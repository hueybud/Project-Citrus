// Copyright 2026 Citrus Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// AIController: ONNX Runtime inference for the Strikers behavioral cloning model.
// Feature extraction mirrors Tools/build_dataset.py::extract_features() exactly.

#include "Core/AIController.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include "Common/Logging/Log.h"
#include "Core/HW/AddressSpace.h"
#include "Core/HW/Memmap.h"
#include "Core/Metadata.h"
#include "InputCommon/GCPadStatus.h"

// ONNX Runtime C++ API — must be included after standard headers
#ifdef _WIN32
// Disable 4251 (DLL interface) warnings from ORT headers on MSVC
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#pragma warning(pop)
#endif

namespace Movie
{

// ---------------------------------------------------------------------------
// Action state vocabs (must stay in sync with build_dataset.py)
// ---------------------------------------------------------------------------

static const std::array<uint32_t, 29> STRIKER_VOCAB = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0xFFFFFFFF,
};
static const std::array<uint32_t, 26> GOALIE_VOCAB = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
    0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19,
};

static constexpr int STRIKER_STATE_DIM = 30;  // 29 known + 1 other
static constexpr int GOALIE_STATE_DIM  = 27;  // 26 known + 1 other
static constexpr int EFFECT_DIM        = 5;
static constexpr int SPEED_ITEM_DIM    = 3;
static constexpr int POWERUP_DIM       = 10;

static constexpr int BUTTON_DIM  = 6;
static constexpr int STICK_DIM   = 4;

// ---------------------------------------------------------------------------
// One-hot helpers
// ---------------------------------------------------------------------------

static void AppendOneHot(std::vector<float>& out, int idx, int dim)
{
  int start = static_cast<int>(out.size());
  out.resize(start + dim, 0.0f);
  if (idx >= 0 && idx < dim)
    out[start + idx] = 1.0f;
}

static void AppendStrikerStateOH(std::vector<float>& out, uint32_t state)
{
  int idx = STRIKER_STATE_DIM - 1;  // default: "other" bucket
  for (int i = 0; i < static_cast<int>(STRIKER_VOCAB.size()); i++)
  {
    if (STRIKER_VOCAB[i] == state)
    {
      idx = i;
      break;
    }
  }
  AppendOneHot(out, idx, STRIKER_STATE_DIM);
}

static void AppendGoalieStateOH(std::vector<float>& out, uint32_t state)
{
  int idx = GOALIE_STATE_DIM - 1;
  for (int i = 0; i < static_cast<int>(GOALIE_VOCAB.size()); i++)
  {
    if (GOALIE_VOCAB[i] == state)
    {
      idx = i;
      break;
    }
  }
  AppendOneHot(out, idx, GOALIE_STATE_DIM);
}

static void AppendEffectOH(std::vector<float>& out, int effect_type)
{
  int idx = std::min(std::max(effect_type, 0), EFFECT_DIM - 1);
  AppendOneHot(out, idx, EFFECT_DIM);
}

static void AppendSpeedItemOH(std::vector<float>& out, int speed_item_type)
{
  // 0=none → 0, 7=mushroom → 1, 8=star → 2
  int idx = 0;
  if (speed_item_type == 7)
    idx = 1;
  else if (speed_item_type == 8)
    idx = 2;
  AppendOneHot(out, idx, SPEED_ITEM_DIM);
}

static void AppendPowerupOH(std::vector<float>& out, int32_t ptype)
{
  // -1=empty → 9, 0-8 → 0-8
  int idx = (ptype == -1) ? 9 : std::min(std::max(static_cast<int>(ptype), 0), 8);
  AppendOneHot(out, idx, POWERUP_DIM);
}

static std::pair<float, float> HeadingSinCos(uint16_t heading_u16)
{
  static constexpr float TWO_PI = 6.28318530718f;
  float angle = (heading_u16 / 65536.0f) * TWO_PI;
  return {std::sin(angle), std::cos(angle)};
}

// ---------------------------------------------------------------------------
// Per-character memory read helpers
// ---------------------------------------------------------------------------

struct CharData
{
  float    pos_x, pos_y, pos_z;
  uint32_t action_state;
  uint16_t heading;
  int      effect_type;     // 0-4
  int      speed_item_type; // 0, 7, or 8
  int      speed_item_count;
  float    speed_item_timer;
  bool     is_user_controlled;
};

// Slot layout: [0-3] left strikers, [4] left goalie, [5-8] right strikers, [9] right goalie
static constexpr int LEFT_STRIKER_SLOTS[]  = {0, 1, 2, 3};
static constexpr int LEFT_GOALIE_SLOT      = 4;
static constexpr int RIGHT_STRIKER_SLOTS[] = {5, 6, 7, 8};
static constexpr int RIGHT_GOALIE_SLOT     = 9;

static CharData ReadCharacter(const AddressSpace::Accessors* acc, uint32_t char_ptr,
                              bool is_goalie)
{
  CharData d{};
  if (char_ptr == 0)
    return d;

  d.pos_x = acc->ReadF32(char_ptr + 0x18);
  d.pos_y = acc->ReadF32(char_ptr + 0x1C);
  d.pos_z = acc->ReadF32(char_ptr + 0x20);

  // Goalie stores action state at +0x1D4; strikers at +0x1D8
  d.action_state = is_goalie ? Memory::Read_U32(char_ptr + 0x1D4)
                             : Memory::Read_U32(char_ptr + 0x1D8);

  d.heading = Memory::Read_U16(char_ptr + 0x42);

  uint32_t effect_ptr = Memory::Read_U32(char_ptr + 0x11C);
  if (effect_ptr == Metadata::addressEffectFrozen)
    d.effect_type = 1;
  else if (effect_ptr == Metadata::addressEffectOnFire)
    d.effect_type = 2;
  else if (effect_ptr == Metadata::addressEffectStar)
    d.effect_type = 3;
  else if (effect_ptr == Metadata::addressEffectElectrocuted)
    d.effect_type = 4;

  int32_t raw_speed = static_cast<int32_t>(Memory::Read_U32(char_ptr + 0x370));
  d.speed_item_type  = (raw_speed == -1) ? 0 : raw_speed;
  d.speed_item_count = static_cast<int>(Memory::Read_U32(char_ptr + 0x374));
  d.speed_item_timer = acc->ReadF32(char_ptr + 0x36C);

  d.is_user_controlled = (Memory::Read_U32(char_ptr + 0x1C0) != 0);

  return d;
}

// ---------------------------------------------------------------------------
// Inventory slot
// ---------------------------------------------------------------------------

struct InventorySlot
{
  int32_t type;
  int     charge_count;
};

static InventorySlot ReadInventorySlot(uint32_t team_ptr, int slot_index)
{
  // Team inventory at team_ptr+0x44 (slot 0) and team_ptr+0x50 (slot 1)
  // Each slot: s32 type (4), u32 chargeCount (4), u8 isNew (1), pad (3)
  uint32_t offset = 0x44 + static_cast<uint32_t>(slot_index) * 0x0C;
  InventorySlot s{};
  s.type         = static_cast<int32_t>(Memory::Read_U32(team_ptr + offset));
  s.charge_count = static_cast<int>(Memory::Read_U32(team_ptr + offset + 4));
  return s;
}

// ---------------------------------------------------------------------------
// AIController implementation
// ---------------------------------------------------------------------------

AIController::AIController() = default;

AIController::~AIController()
{
  Shutdown();
}

bool AIController::Load(const std::string& onnx_path)
{
  Shutdown();

  try
  {
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "citrus_ai");

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    // ORT on Windows requires a wide string path
    std::wstring wide_path(onnx_path.begin(), onnx_path.end());
    m_session = std::make_unique<Ort::Session>(*m_env, wide_path.c_str(), opts);
#else
    m_session = std::make_unique<Ort::Session>(*m_env, onnx_path.c_str(), opts);
#endif

    // Validate expected I/O: features + h_in + c_in → btn_probs + stick_vals + h_out + c_out
    size_t num_inputs  = m_session->GetInputCount();
    size_t num_outputs = m_session->GetOutputCount();

    if (num_inputs != 3 || num_outputs != 4)
    {
      ERROR_LOG_FMT(CORE,
                    "AIController: unexpected I/O count: {} inputs {} outputs "
                    "(expected 3 inputs: features/h_in/c_in, "
                    "4 outputs: btn_probs/stick_vals/h_out/c_out)",
                    num_inputs, num_outputs);
      Shutdown();
      return false;
    }

    // Check features input shape [1, FEATURE_DIM]
    auto feat_info  = m_session->GetInputTypeInfo(0);
    auto feat_shape = feat_info.GetTensorTypeAndShapeInfo().GetShape();
    if (feat_shape.size() != 2 || feat_shape[1] != FEATURE_DIM)
    {
      ERROR_LOG_FMT(CORE, "AIController: unexpected features shape (expected [1,{}])",
                    FEATURE_DIM);
      Shutdown();
      return false;
    }

    // Check h_in / c_in shape [LSTM_LAYERS, 1, HIDDEN_SIZE]
    for (int i = 1; i <= 2; ++i)
    {
      auto hc_info  = m_session->GetInputTypeInfo(i);
      auto hc_shape = hc_info.GetTensorTypeAndShapeInfo().GetShape();
      if (hc_shape.size() != 3 || hc_shape[0] != LSTM_LAYERS || hc_shape[2] != HIDDEN_SIZE)
      {
        ERROR_LOG_FMT(CORE,
                      "AIController: unexpected h/c input {} shape "
                      "(expected [{},1,{}])",
                      i, LSTM_LAYERS, HIDDEN_SIZE);
        Shutdown();
        return false;
      }
    }

    m_loaded = true;
    INFO_LOG_FMT(CORE, "AIController: loaded model from {}", onnx_path);
    return true;
  }
  catch (const Ort::Exception& e)
  {
    ERROR_LOG_FMT(CORE, "AIController: ORT exception loading model: {}", e.what());
    Shutdown();
    return false;
  }
  catch (const std::exception& e)
  {
    ERROR_LOG_FMT(CORE, "AIController: exception loading model: {}", e.what());
    Shutdown();
    return false;
  }
}

void AIController::Shutdown()
{
  m_session.reset();
  m_env.reset();
  m_loaded  = false;
  m_h_state.fill(0.0f);
  m_c_state.fill(0.0f);
  std::memset(&m_last_output, 0, sizeof(m_last_output));
  m_last_output.stickX      = 128;
  m_last_output.stickY      = 128;
  m_last_output.substickX   = 128;
  m_last_output.substickY   = 128;
  m_last_output.isConnected = true;
}

bool AIController::IsLoaded() const
{
  return m_loaded;
}

void AIController::OnFrameEnd(int controlled_port, bool mirror_x)
{
  if (!m_loaded || !m_session)
    return;

  std::vector<float> features = ReadGameState(controlled_port, mirror_x);
  if (features.empty())
  {
    // Not in active play (menus, goal celebration, etc.) — reset LSTM state so the
    // next active phase starts with a clean context.
    m_match_active = false;
    m_h_state.fill(0.0f);
    m_c_state.fill(0.0f);
    return;
  }

  try
  {
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Input 0: features [1, FEATURE_DIM]
    int64_t feat_shape[] = {1, FEATURE_DIM};
    Ort::Value feat_tensor = Ort::Value::CreateTensor<float>(
        mem_info, features.data(), FEATURE_DIM, feat_shape, 2);

    // Inputs 1 & 2: h_in, c_in — [LSTM_LAYERS, 1, HIDDEN_SIZE]
    int64_t hc_shape[] = {LSTM_LAYERS, 1, HIDDEN_SIZE};
    Ort::Value h_tensor = Ort::Value::CreateTensor<float>(
        mem_info, m_h_state.data(), HC_SIZE, hc_shape, 3);
    Ort::Value c_tensor = Ort::Value::CreateTensor<float>(
        mem_info, m_c_state.data(), HC_SIZE, hc_shape, 3);

    std::array<Ort::Value, 3> inputs = {
        std::move(feat_tensor), std::move(h_tensor), std::move(c_tensor)};

    const char* input_names[]  = {"features", "h_in", "c_in"};
    const char* output_names[] = {"btn_probs", "stick_vals", "h_out", "c_out"};

    auto outputs = m_session->Run(Ort::RunOptions{nullptr},
                                  input_names, inputs.data(), 3,
                                  output_names, 4);

    const float* btn_probs  = outputs[0].GetTensorData<float>();
    const float* stick_vals = outputs[1].GetTensorData<float>();

    m_last_output  = DecodeOutput(btn_probs, stick_vals, mirror_x);
    m_match_active = true;

    // Carry LSTM state forward to the next frame
    const float* h_out = outputs[2].GetTensorData<float>();
    const float* c_out = outputs[3].GetTensorData<float>();
    std::copy(h_out, h_out + HC_SIZE, m_h_state.begin());
    std::copy(c_out, c_out + HC_SIZE, m_c_state.begin());
  }
  catch (const Ort::Exception& e)
  {
    WARN_LOG_FMT(CORE, "AIController: ORT inference error: {}", e.what());
    m_match_active = false;
  }
}

bool AIController::IsMatchActive() const
{
  return m_match_active;
}

GCPadStatus AIController::GetLastOutput() const
{
  return m_last_output;
}

// ---------------------------------------------------------------------------
// Feature extraction — mirrors build_dataset.py::extract_features() exactly
// ---------------------------------------------------------------------------

std::vector<float> AIController::ReadGameState(int controlled_port, bool mirror_x) const
{
  const AddressSpace::Accessors* acc =
      AddressSpace::GetAccessors(AddressSpace::Type::Effective);

  const float sign = mirror_x ? -1.0f : 1.0f;
  auto mx = [sign](float x) { return sign * x; };

  // Read cGame pointer and game phase
  constexpr uint32_t CGAME_SINGLETON = 0x80373708;
  uint32_t cGamePtr = Memory::Read_U32(CGAME_SINGLETON);
  if (cGamePtr == 0)
  {
    INFO_LOG_FMT(CORE, "AIController::ReadGameState: cGamePtr is null");
    return {};  // game not loaded
  }

  uint32_t game_phase = Memory::Read_U32(cGamePtr + 0x24);
  // Log once per ~60 frames so we can see the phase without spamming
  static uint32_t s_log_counter = 0;
  if (++s_log_counter % 60 == 1)
  {
    INFO_LOG_FMT(CORE, "AIController::ReadGameState: cGamePtr=0x{:08X} game_phase={}", cGamePtr,
                 game_phase);
  }
  if (game_phase != 4 && game_phase != 5)
    return {};  // not active play — skip inference

  // Read all 10 character pointers
  // Slot layout: [0-3] left strikers, [4] left goalie, [5-8] right strikers, [9] right goalie
  uint32_t char_ptrs[10] = {};
  for (int i = 0; i < 4; i++)
    char_ptrs[i]     = Memory::Read_U32(Metadata::addressCharacterPointersBase + i * 4);
  char_ptrs[4]       = Memory::Read_U32(Metadata::addressLeftGoaliePointer);
  for (int i = 0; i < 4; i++)
    char_ptrs[5 + i] = Memory::Read_U32(Metadata::addressCharacterPointersBase + 0x10 + i * 4);
  char_ptrs[9]       = Memory::Read_U32(Metadata::addressRightGoaliePointer);

  // Read character structs
  CharData chars[10] = {};
  for (int i = 0; i < 4; i++)
    chars[i]     = ReadCharacter(acc, char_ptrs[i],     false);
  chars[4]       = ReadCharacter(acc, char_ptrs[4],     true);
  for (int i = 0; i < 4; i++)
    chars[5 + i] = ReadCharacter(acc, char_ptrs[5 + i], false);
  chars[9]       = ReadCharacter(acc, char_ptrs[9],     true);

  // Determine subject team and striker slots from mirror_x
  // mirror_x=false → LEFT team (slots 0-3, goalie 4)
  // mirror_x=true  → RIGHT team (slots 5-8, goalie 9)
  const int* own_striker_slots   = mirror_x ? RIGHT_STRIKER_SLOTS : LEFT_STRIKER_SLOTS;
  const int  own_goalie_slot     = mirror_x ? RIGHT_GOALIE_SLOT   : LEFT_GOALIE_SLOT;
  const int* enemy_striker_slots = mirror_x ? LEFT_STRIKER_SLOTS  : RIGHT_STRIKER_SLOTS;
  const int  enemy_goalie_slot   = mirror_x ? LEFT_GOALIE_SLOT    : RIGHT_GOALIE_SLOT;

  // Find self slot: the user-controlled striker on our team.
  // Falls back to first slot of team (can briefly be uncontrolled at kickoff boundary).
  int self_slot = own_striker_slots[0];
  for (int i = 0; i < 4; i++)
  {
    int s = own_striker_slots[i];
    if (chars[s].is_user_controlled)
    {
      self_slot = s;
      break;
    }
  }

  // Ball state
  float bpx = mx(acc->ReadF32(Metadata::addressBallXPos));
  float bpy = acc->ReadF32(Metadata::addressBallYPos);
  float bpz = acc->ReadF32(Metadata::addressBallZPos);

  uint32_t ball_ptr = Memory::Read_U32(Metadata::addressBallPointer);
  float bvx = 0.0f, bvy = 0.0f, bvz = 0.0f;
  uint32_t ball_owner_ptr = 0;
  bool is_perfect_pass    = false;
  if (ball_ptr != 0)
  {
    bvx           = mx(acc->ReadF32(ball_ptr + 0x58));
    bvy           =    acc->ReadF32(ball_ptr + 0x5C);
    bvz           =    acc->ReadF32(ball_ptr + 0x60);
    ball_owner_ptr = Memory::Read_U32(ball_ptr + 0x24);
    is_perfect_pass = (Memory::Read_U8(ball_ptr + 0xA1) != 0);
  }

  float ball_charge = static_cast<float>(Memory::Read_U32(Metadata::addressChargedBallAmount));

  // Resolve ball owner slot (0-9 or 10 = none)
  int owner_slot = 10;
  for (int i = 0; i < 10; i++)
  {
    if (char_ptrs[i] != 0 && char_ptrs[i] == ball_owner_ptr)
    {
      owner_slot = i;
      break;
    }
  }

  // Goal line X (absolute after canonical mirror sign is +)
  float goal_x = std::abs(acc->ReadF32(0x802a3e60));

  // Team pointers for inventory
  uint32_t left_team_ptr  = Memory::Read_U32(Metadata::addressTeam1Pointer);
  uint32_t right_team_ptr = Memory::Read_U32(Metadata::addressTeam2Pointer);
  uint32_t own_team_ptr   = mirror_x ? right_team_ptr : left_team_ptr;
  uint32_t enemy_team_ptr = mirror_x ? left_team_ptr  : right_team_ptr;

  // Score and time
  int left_score  = static_cast<int>(Memory::Read_U16(Metadata::addressLeftSideScore));
  int right_score = static_cast<int>(Memory::Read_U16(Metadata::addressRightSideScore));
  int score_diff  = left_score - right_score;
  if (mirror_x)
    score_diff = -score_diff;

  float game_time = acc->ReadF32(Metadata::addressTimeElapsed);
  float match_time_allotted = static_cast<float>(
      std::max(Memory::Read_U32(Metadata::addressMatchTimeAllotted), 1u));

  // ---------------------------------------------------------------------------
  // Build feature vector
  // ---------------------------------------------------------------------------
  std::vector<float> feat;
  feat.reserve(FEATURE_DIM);

  // --- 1. Ball (8) ---
  feat.push_back(bpx);
  feat.push_back(bpy);
  feat.push_back(bpz);
  feat.push_back(bvx);
  feat.push_back(bvy);
  feat.push_back(bvz);
  feat.push_back(ball_charge / 35.0f);
  feat.push_back(is_perfect_pass ? 1.0f : 0.0f);

  // --- 2. Ball owner one-hot (11) ---
  AppendOneHot(feat, owner_slot, 11);

  // --- 3. Self character (48) ---
  {
    const CharData& sc = chars[self_slot];
    float sc_px = mx(sc.pos_x);
    float sc_py = sc.pos_y;

    float dx   = goal_x - sc_px;
    float dy   = sc_py;
    float dist = std::sqrt(dx * dx + dy * dy) + 1e-6f;

    auto [sin_h, cos_h] = HeadingSinCos(sc.heading);
    if (mirror_x)
      cos_h = -cos_h;

    // Δpos to ball (3)
    feat.push_back(sc_px - bpx);
    feat.push_back(sc_py - bpy);
    feat.push_back(sc.pos_z);

    // State one-hot (30)
    AppendStrikerStateOH(feat, sc.action_state);

    // Heading (2)
    feat.push_back(sin_h);
    feat.push_back(cos_h);

    // Goal dist+angle (3)
    feat.push_back(dist / 40.0f);
    feat.push_back(dx / dist);
    feat.push_back(dy / dist);

    // Effect (5)
    AppendEffectOH(feat, sc.effect_type);

    // Speed item (3)
    AppendSpeedItemOH(feat, sc.speed_item_type);

    // Item timer (1)
    feat.push_back(sc.speed_item_timer / 10.0f);

    // Is ball carrier (1)
    feat.push_back(self_slot == owner_slot ? 1.0f : 0.0f);
  }

  // --- 4. 3 other friendly strikers, sorted by distance to ball (3 × 36 = 108) ---
  {
    // Collect other friendly strikers (excluding self)
    struct SlotDist
    {
      int   slot;
      float dist2;
    };
    SlotDist others[3];
    int n_others = 0;
    for (int i = 0; i < 4 && n_others < 3; i++)
    {
      int s = own_striker_slots[i];
      if (s == self_slot)
        continue;
      float dx2 = mx(chars[s].pos_x) - bpx;
      float dy2 = chars[s].pos_y - bpy;
      others[n_others++] = {s, dx2 * dx2 + dy2 * dy2};
    }
    std::sort(others, others + n_others,
              [](const SlotDist& a, const SlotDist& b) { return a.dist2 < b.dist2; });

    for (int k = 0; k < 3; k++)
    {
      if (k < n_others)
      {
        int s = others[k].slot;
        const CharData& ch = chars[s];
        float ch_px = mx(ch.pos_x);
        auto [sh, ch_cos] = HeadingSinCos(ch.heading);
        if (mirror_x)
          ch_cos = -ch_cos;

        feat.push_back(ch_px - bpx);          // Δpos (3)
        feat.push_back(ch.pos_y - bpy);
        feat.push_back(ch.pos_z);
        AppendStrikerStateOH(feat, ch.action_state);  // state (30)
        feat.push_back(sh);                           // heading (2)
        feat.push_back(ch_cos);
        feat.push_back(s == owner_slot ? 1.0f : 0.0f);  // ball carrier (1)
      }
      else
      {
        // Zero-pad for missing slot (shouldn't occur with full 4-striker teams)
        for (int z = 0; z < 36; z++)
          feat.push_back(0.0f);
      }
    }
  }

  // --- 5. Friendly goalie (30) ---
  {
    const CharData& gs = chars[own_goalie_slot];
    feat.push_back(mx(gs.pos_x) - bpx);
    feat.push_back(gs.pos_y - bpy);
    feat.push_back(gs.pos_z);
    AppendGoalieStateOH(feat, gs.action_state);
  }

  // --- 6. 4 enemy strikers, sorted by distance to ball (4 × 36 = 144) ---
  {
    struct SlotDist
    {
      int   slot;
      float dist2;
    };
    SlotDist enemies[4];
    for (int i = 0; i < 4; i++)
    {
      int s = enemy_striker_slots[i];
      float dx2 = mx(chars[s].pos_x) - bpx;
      float dy2 = chars[s].pos_y - bpy;
      enemies[i] = {s, dx2 * dx2 + dy2 * dy2};
    }
    std::sort(enemies, enemies + 4,
              [](const SlotDist& a, const SlotDist& b) { return a.dist2 < b.dist2; });

    for (int k = 0; k < 4; k++)
    {
      int s = enemies[k].slot;
      const CharData& ch = chars[s];
      float ch_px = mx(ch.pos_x);
      auto [sh, ch_cos] = HeadingSinCos(ch.heading);
      if (mirror_x)
        ch_cos = -ch_cos;

      feat.push_back(ch_px - bpx);
      feat.push_back(ch.pos_y - bpy);
      feat.push_back(ch.pos_z);
      AppendStrikerStateOH(feat, ch.action_state);
      feat.push_back(sh);
      feat.push_back(ch_cos);
      feat.push_back(s == owner_slot ? 1.0f : 0.0f);
    }
  }

  // --- 7. Enemy goalie (30) ---
  {
    const CharData& eg = chars[enemy_goalie_slot];
    feat.push_back(mx(eg.pos_x) - bpx);
    feat.push_back(eg.pos_y - bpy);
    feat.push_back(eg.pos_z);
    AppendGoalieStateOH(feat, eg.action_state);
  }

  // --- 8. Own inventory (2 × 11 = 22) ---
  if (own_team_ptr != 0)
  {
    for (int s = 0; s < 2; s++)
    {
      InventorySlot inv = ReadInventorySlot(own_team_ptr, s);
      AppendPowerupOH(feat, inv.type);
      feat.push_back(inv.charge_count / 5.0f);
    }
  }
  else
  {
    for (int z = 0; z < 22; z++)
      feat.push_back(0.0f);
  }

  // --- 9. Enemy inventory (2 × 11 = 22) ---
  if (enemy_team_ptr != 0)
  {
    for (int s = 0; s < 2; s++)
    {
      InventorySlot inv = ReadInventorySlot(enemy_team_ptr, s);
      AppendPowerupOH(feat, inv.type);
      feat.push_back(inv.charge_count / 5.0f);
    }
  }
  else
  {
    for (int z = 0; z < 22; z++)
      feat.push_back(0.0f);
  }

  // --- 10. Tactical summary (5) ---
  {
    float sc_px_ts         = mx(chars[self_slot].pos_x);
    float sc_py_ts         = chars[self_slot].pos_y;
    float sdx              = sc_px_ts - bpx;
    float sdy              = sc_py_ts - bpy;
    float self_dist_to_ball = std::sqrt(sdx * sdx + sdy * sdy);

    int self_rank = 0;
    for (int i = 0; i < 4; i++)
    {
      int s = own_striker_slots[i];
      if (s == self_slot)
        continue;
      float dx2 = mx(chars[s].pos_x) - bpx;
      float dy2 = chars[s].pos_y - bpy;
      if (dx2 * dx2 + dy2 * dy2 < self_dist_to_ball * self_dist_to_ball)
        self_rank++;
    }
    feat.push_back(self_dist_to_ball / 40.0f);
    feat.push_back(static_cast<float>(self_rank) / 3.0f);
    feat.push_back(self_rank == 0 ? 1.0f : 0.0f);

    float min_en_dist2 = 1e18f;
    int   nearest_en   = enemy_striker_slots[0];
    for (int i = 0; i < 4; i++)
    {
      int   s  = enemy_striker_slots[i];
      float ex = mx(chars[s].pos_x) - sc_px_ts;
      float ey = chars[s].pos_y - sc_py_ts;
      float d2 = ex * ex + ey * ey;
      if (d2 < min_en_dist2)
      {
        min_en_dist2 = d2;
        nearest_en   = s;
      }
    }
    float nearest_enemy_dist = std::sqrt(min_en_dist2);
    feat.push_back(nearest_enemy_dist / 40.0f);

    float goal_dx  = goal_x - sc_px_ts;
    float goal_dy  = 0.0f - sc_py_ts;
    float goal_len = std::sqrt(goal_dx * goal_dx + goal_dy * goal_dy) + 1e-6f;
    float en_dx    = mx(chars[nearest_en].pos_x) - sc_px_ts;
    float en_dy    = chars[nearest_en].pos_y - sc_py_ts;
    float en_len   = nearest_enemy_dist + 1e-6f;
    feat.push_back((goal_dx / goal_len) * (en_dx / en_len) + (goal_dy / goal_len) * (en_dy / en_len));
  }

  // --- 11. Score diff + time fraction (2) ---
  float score_diff_clamped = std::max(-5.0f, std::min(5.0f, static_cast<float>(score_diff)));
  feat.push_back(score_diff_clamped / 5.0f);
  feat.push_back(std::min(game_time / match_time_allotted, 1.0f));

  if (static_cast<int>(feat.size()) != FEATURE_DIM)
  {
    ERROR_LOG_FMT(CORE, "AIController: feature dim mismatch: got {} expected {}",
                  feat.size(), FEATURE_DIM);
    return {};
  }

  return feat;
}

// ---------------------------------------------------------------------------
// DecodeOutput
// ---------------------------------------------------------------------------

GCPadStatus AIController::DecodeOutput(const float* btn_probs, const float* stick_vals,
                                        bool mirror_x) const
{
  GCPadStatus pad{};
  pad.isConnected = true;
  pad.button      = PAD_USE_ORIGIN;

  // Buttons: probabilities → binary threshold at 0.5
  // Labels: 0=A 1=B 2=X 3=Y 4=L 5=R
  if (btn_probs[0] > 0.5f) pad.button |= PAD_BUTTON_A;
  if (btn_probs[1] > 0.5f) pad.button |= PAD_BUTTON_B;
  if (btn_probs[2] > 0.5f) pad.button |= PAD_BUTTON_X;
  if (btn_probs[3] > 0.5f) pad.button |= PAD_BUTTON_Y;
  if (btn_probs[4] > 0.5f) pad.button |= PAD_TRIGGER_L;
  if (btn_probs[5] > 0.5f) pad.button |= PAD_TRIGGER_R;

  // analogA/B expected by Dolphin SI layer
  if (btn_probs[0] > 0.5f) pad.analogA = 0xFF;
  if (btn_probs[1] > 0.5f) pad.analogB = 0xFF;

  // Trigger bytes (full press when digital bit set)
  if (btn_probs[4] > 0.5f) pad.triggerLeft  = 0xFF;
  if (btn_probs[5] > 0.5f) pad.triggerRight = 0xFF;

  // Sticks: tanh output in [-1, 1]; label encoding: (stick_byte - 128) / 128
  // Invert: stick_byte = clamp(val * 128 + 128, 0, 255)
  // Mirror: if mirror_x was applied during feature extraction, the model learned
  // the mirrored stick_x. Negate to convert back to native GC orientation.
  auto DecodeStick = [](float val) -> uint8_t
  {
    float raw = val * 128.0f + 128.0f;
    return static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, raw)));
  };

  float stick_x  = stick_vals[0];
  float stick_y  = stick_vals[1];
  float cstick_x = stick_vals[2];
  float cstick_y = stick_vals[3];

  if (mirror_x)
  {
    stick_x  = -stick_x;
    cstick_x = -cstick_x;
  }

  pad.stickX    = DecodeStick(stick_x);
  pad.stickY    = DecodeStick(stick_y);
  pad.substickX = DecodeStick(cstick_x);
  pad.substickY = DecodeStick(cstick_y);

  return pad;
}

}  // namespace Movie

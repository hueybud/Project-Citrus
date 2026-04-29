// Copyright 2026 Citrus Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// AIController: ONNX Runtime inference for the CitrusTransformerBC model (v6).
// Feature extraction mirrors SMS AI/scripts/build_dataset.py::extract_features() exactly.
// KV cache carries temporal context across frames within a play segment.

#include "Core/AIController.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include "Common/Logging/Log.h"
#include "Core/HW/AddressSpace.h"
#include "Core/HW/Memmap.h"
#include "Core/Metadata.h"
#include "InputCommon/GCPadStatus.h"

// ONNX Runtime C++ API
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4251)
#endif
#include <onnxruntime_cxx_api.h>
#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef _WIN32
// For SetThreadPriority on the inference worker.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace Movie
{

// ---------------------------------------------------------------------------
// Windowed timing stats — percentile dump of recent samples.
// Used to characterize tail latency (p95/p99/max) which sample-every-N logs
// miss but which directly causes FPS drops.  Single-thread access only.
// ---------------------------------------------------------------------------
namespace
{
struct WindowedStats
{
  static constexpr size_t WINDOW = 600;  // ~10s @ 60Hz
  std::array<float, WINDOW> samples{};
  size_t n            = 0;
  int    spikes       = 0;
  float  worst        = 0.0f;
  float  threshold_ms = 10.0f;  // tunable per metric (e.g. 17ms for interval)

  explicit WindowedStats(float threshold = 10.0f) : threshold_ms(threshold) {}

  void Push(float v)
  {
    if (n < WINDOW)
      samples[n] = v;
    n++;
    if (v > threshold_ms)
      spikes++;
    if (v > worst)
      worst = v;
  }

  bool Full() const { return n >= WINDOW; }

  void Dump(const char* label) const
  {
    if (n == 0)
      return;
    const size_t count = std::min(n, WINDOW);
    std::array<float, WINDOW> sorted = samples;
    std::sort(sorted.begin(), sorted.begin() + count);
    const float sum  = std::accumulate(sorted.begin(), sorted.begin() + count, 0.0f);
    const float mean = sum / static_cast<float>(count);
    auto pct = [&](float p) {
      const size_t idx = static_cast<size_t>(p * static_cast<float>(count - 1));
      return sorted[idx];
    };
    INFO_LOG_FMT(CORE,
                 "AI stats {}: n={} mean={:.2f} p50={:.2f} p95={:.2f} p99={:.2f} "
                 "max={:.2f} spikes>{:.0f}ms={}",
                 label, count, mean, pct(0.50f), pct(0.95f), pct(0.99f),
                 worst, threshold_ms, spikes);
  }

  void Reset()
  {
    n      = 0;
    spikes = 0;
    worst  = 0.0f;
  }
};
}  // namespace

// Stochastic sampling RNG for button presses (Bernoulli policy).

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
static constexpr int FIELD_ITEM_K      = 4;   // top-K nearest items
static constexpr int FIELD_ITEM_DIM    = 8;   // type(1) + Δpos(3) + vel(3) + strength(1)

// ---------------------------------------------------------------------------
// v6 helpers — integer index (for model embedding) instead of one-hot
// ---------------------------------------------------------------------------

static float StrikerStateIdx(uint32_t state)
{
  for (int i = 0; i < static_cast<int>(STRIKER_VOCAB.size()); i++)
  {
    if (STRIKER_VOCAB[i] == state)
      return static_cast<float>(i);
  }
  return static_cast<float>(STRIKER_STATE_DIM - 1);  // unknown → last bucket
}

static float GoalieStateIdx(uint32_t state)
{
  for (int i = 0; i < static_cast<int>(GOALIE_VOCAB.size()); i++)
  {
    if (GOALIE_VOCAB[i] == state)
      return static_cast<float>(i);
  }
  return static_cast<float>(GOALIE_STATE_DIM - 1);  // unknown → last bucket
}

// One-hot helpers (still used for effect, speed_item, powerup, ball_owner)
static void AppendOneHot(std::vector<float>& out, int idx, int dim)
{
  int start = static_cast<int>(out.size());
  out.resize(start + dim, 0.0f);
  if (idx >= 0 && idx < dim)
    out[start + idx] = 1.0f;
}

static void AppendEffectOH(std::vector<float>& out, int effect_type)
{
  AppendOneHot(out, std::min(std::max(effect_type, 0), EFFECT_DIM - 1), EFFECT_DIM);
}

static void AppendSpeedItemOH(std::vector<float>& out, int speed_item_type)
{
  int idx = 0;
  if (speed_item_type == 7)
    idx = 1;
  else if (speed_item_type == 8)
    idx = 2;
  AppendOneHot(out, idx, SPEED_ITEM_DIM);
}

static void AppendPowerupOH(std::vector<float>& out, int32_t ptype)
{
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
  int      effect_type;
  int      speed_item_type;
  float    speed_item_timer;
  bool     is_user_controlled;
};

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

  int32_t raw_speed  = static_cast<int32_t>(Memory::Read_U32(char_ptr + 0x370));
  d.speed_item_type  = (raw_speed == -1) ? 0 : raw_speed;
  d.speed_item_timer = acc->ReadF32(char_ptr + 0x36C);

  d.is_user_controlled = (Memory::Read_U32(char_ptr + 0x1C0) != 0);

  return d;
}

// ---------------------------------------------------------------------------
// Field item read helper (mirrors GameStateFrame.cpp ReadItems)
// ---------------------------------------------------------------------------

struct FieldItem
{
  float pos_x, pos_y, pos_z;
  float vel_x, vel_y, vel_z;
  int   powerup_type;
  int   strength_level;
};

static constexpr int MAX_ACTIVE_ITEMS = 25;

static std::vector<FieldItem> ReadFieldItems(const AddressSpace::Accessors* acc)
{
  std::vector<FieldItem> items;
  for (int i = 0; i < MAX_ACTIVE_ITEMS; i++)
  {
    uint32_t item_ptr = Memory::Read_U32(
        Metadata::addressActivePowerupArray + static_cast<uint32_t>(i) * 4);
    if (item_ptr == 0)
      continue;

    FieldItem fi{};
    fi.pos_x          = acc->ReadF32(item_ptr + 0x2C);
    fi.pos_y          = acc->ReadF32(item_ptr + 0x30);
    fi.pos_z          = acc->ReadF32(item_ptr + 0x34);
    fi.vel_x          = acc->ReadF32(item_ptr + 0x44);
    fi.vel_y          = acc->ReadF32(item_ptr + 0x48);
    fi.vel_z          = acc->ReadF32(item_ptr + 0x4C);
    fi.powerup_type   = static_cast<int>(Memory::Read_U32(item_ptr + 0x18));
    fi.strength_level = static_cast<int>(Memory::Read_U32(item_ptr + 0x6C));
    items.push_back(fi);
  }
  return items;
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
  uint32_t offset = 0x44 + static_cast<uint32_t>(slot_index) * 0x0C;
  InventorySlot s{};
  s.type         = static_cast<int32_t>(Memory::Read_U32(team_ptr + offset));
  s.charge_count = static_cast<int>(Memory::Read_U32(team_ptr + offset + 4));
  return s;
}

// ---------------------------------------------------------------------------
// LocalOnnxBackend — runs ONNX Runtime on a dedicated worker thread.
// Owns: ORT session, KV cache, prev_labels, output slot.
// All public methods are thread-safe per the AIInferenceBackend contract.
// ---------------------------------------------------------------------------

namespace
{
class LocalOnnxBackend : public AIInferenceBackend
{
public:
  LocalOnnxBackend() = default;
  ~LocalOnnxBackend() override { Shutdown(); }

  bool Load(const std::string& onnx_path);

  // AIInferenceBackend
  void        Submit(AIInputFrame frame) override;
  GCPadStatus GetLastOutput() const override;
  bool        HasOutput() const override;
  void        Shutdown() override;

private:
  void        WorkerLoop();
  GCPadStatus DecodeOutput(const float* btn_probs, const float* stick_vals,
                           bool mirror_x);
  std::vector<float> BuildFullFeatures(const std::vector<float>& core) const;

  // ORT session — touched only by Load()/Shutdown() (emu thread, before/after
  // worker is running) and by WorkerLoop() (worker thread, between).  The
  // worker join in Shutdown() guarantees no concurrent access to the session.
  std::unique_ptr<Ort::Env>     m_env;
  std::unique_ptr<Ort::Session> m_session;

  // Worker thread + input slot (single-slot, latest-wins).
  std::thread                 m_worker;
  mutable std::mutex          m_in_mu;
  std::condition_variable     m_in_cv;
  std::optional<AIInputFrame> m_pending;
  bool                        m_stop = false;

  // Output slot (mutex — GCPadStatus is 11 bytes, not atomic-friendly).
  mutable std::mutex m_out_mu;
  GCPadStatus        m_last_output{};
  bool               m_has_output = false;

  // Worker-owned context.  No synchronization; only the worker touches these.
  std::array<float, AIModelDims::KV_CACHE_SIZE>   m_kv_cache{};
  std::array<float, AIModelDims::PREV_ACTION_DIM> m_prev_labels{};
};

bool LocalOnnxBackend::Load(const std::string& onnx_path)
{
  Shutdown();

  try
  {
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "citrus_ai");

    Ort::SessionOptions opts;
    // intra-op = 2: lets ORT parallelize matmul across 2 threads internally.
    // Python benchmark on this model shows intra=2 gives both lower mean
    // (4.40 vs 4.66) and tighter p99 (6.51 vs 8.09) than intra=1.  Costs one
    // extra core during the ~5ms inference window per frame — fine on any
    // modern CPU.  inter-op left at default since the graph is sequential.
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    std::wstring wide_path(onnx_path.begin(), onnx_path.end());
    m_session = std::make_unique<Ort::Session>(*m_env, wide_path.c_str(), opts);
#else
    m_session = std::make_unique<Ort::Session>(*m_env, onnx_path.c_str(), opts);
#endif

    // Validate expected I/O (v6):
    //   inputs:  features [1, 194], kv_cache_in [3, 2, 1, 127, 512]
    //   outputs: btn_probs [1, 7], stick_vals [1, 4], kv_cache_out [3, 2, 1, 127, 512]
    size_t num_inputs  = m_session->GetInputCount();
    size_t num_outputs = m_session->GetOutputCount();

    if (num_inputs != 2 || num_outputs != 3)
    {
      ERROR_LOG_FMT(CORE,
                    "LocalOnnxBackend: unexpected I/O count: {} inputs {} outputs "
                    "(expected 2 inputs: features/kv_cache_in, "
                    "3 outputs: btn_probs/stick_vals/kv_cache_out)",
                    num_inputs, num_outputs);
      Shutdown();
      return false;
    }

    auto feat_info  = m_session->GetInputTypeInfo(0);
    auto feat_shape = feat_info.GetTensorTypeAndShapeInfo().GetShape();
    if (feat_shape.size() != 2 || feat_shape[1] != AIModelDims::FEATURE_DIM)
    {
      ERROR_LOG_FMT(CORE, "LocalOnnxBackend: unexpected features shape (expected [1,{}])",
                    AIModelDims::FEATURE_DIM);
      Shutdown();
      return false;
    }

    auto kv_info  = m_session->GetInputTypeInfo(1);
    auto kv_shape = kv_info.GetTensorTypeAndShapeInfo().GetShape();
    if (kv_shape.size() != 5 ||
        kv_shape[0] != AIModelDims::KV_CACHE_LAYERS || kv_shape[1] != 2 ||
        kv_shape[3] != AIModelDims::KV_CACHE_SEQ    || kv_shape[4] != AIModelDims::KV_CACHE_DIM)
    {
      ERROR_LOG_FMT(CORE,
                    "LocalOnnxBackend: unexpected kv_cache_in shape (expected [{},2,1,{},{}])",
                    AIModelDims::KV_CACHE_LAYERS, AIModelDims::KV_CACHE_SEQ,
                    AIModelDims::KV_CACHE_DIM);
      Shutdown();
      return false;
    }

    INFO_LOG_FMT(CORE, "LocalOnnxBackend: loaded v6 transformer model from {}", onnx_path);

    // Spin up the worker now that the session is valid.
    m_stop   = false;
    m_worker = std::thread(&LocalOnnxBackend::WorkerLoop, this);
    return true;
  }
  catch (const Ort::Exception& e)
  {
    ERROR_LOG_FMT(CORE, "LocalOnnxBackend: ORT exception loading model: {}", e.what());
    Shutdown();
    return false;
  }
  catch (const std::exception& e)
  {
    ERROR_LOG_FMT(CORE, "LocalOnnxBackend: exception loading model: {}", e.what());
    Shutdown();
    return false;
  }
}

void LocalOnnxBackend::Submit(AIInputFrame frame)
{
  {
    std::lock_guard<std::mutex> lk(m_in_mu);
    m_pending = std::move(frame);  // overwrites any unread previous frame
  }
  m_in_cv.notify_one();
}

GCPadStatus LocalOnnxBackend::GetLastOutput() const
{
  std::lock_guard<std::mutex> lk(m_out_mu);
  return m_last_output;
}

bool LocalOnnxBackend::HasOutput() const
{
  std::lock_guard<std::mutex> lk(m_out_mu);
  return m_has_output;
}

void LocalOnnxBackend::Shutdown()
{
  {
    std::lock_guard<std::mutex> lk(m_in_mu);
    m_stop = true;
    m_pending.reset();
  }
  m_in_cv.notify_all();
  if (m_worker.joinable())
    m_worker.join();

  // Worker is now gone; safe to release the session.
  m_session.reset();
  m_env.reset();

  // Reset output to a neutral pad so any late GetLastOutput() reads after
  // a model unload return safe values.
  std::lock_guard<std::mutex> lk(m_out_mu);
  m_has_output = false;
  std::memset(&m_last_output, 0, sizeof(m_last_output));
  m_last_output.stickX      = 0x80;
  m_last_output.stickY      = 0x80;
  m_last_output.substickX   = 0x80;
  m_last_output.substickY   = 0x80;
  m_last_output.isConnected = true;

  // Worker-owned state — worker is gone, safe to clear.
  m_kv_cache.fill(0.0f);
  m_prev_labels.fill(0.0f);
}

std::vector<float> LocalOnnxBackend::BuildFullFeatures(const std::vector<float>& core) const
{
  std::vector<float> full;
  full.reserve(AIModelDims::FEATURE_DIM);
  full.insert(full.end(), core.begin(), core.end());
  full.insert(full.end(), m_prev_labels.begin(), m_prev_labels.end());
  return full;
}

void LocalOnnxBackend::WorkerLoop()
{
#ifdef _WIN32
  // Without this bump the worker runs at NORMAL priority and is constantly
  // preempted by Dolphin's emu thread mid-Run(), doubling per-inference wall
  // time (5.5ms → 11ms measured).  ABOVE_NORMAL is enough to outrank emu.
  ::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif

  // Worker-thread windowed timing buckets (independent from emu-thread stats).
  WindowedStats stats_infer(10.0f);   // >10ms = potential FPS pressure (legacy)
  int           feat_log_counter  = 0;
  int           infer_log_counter = 0;

  while (true)
  {
    AIInputFrame frame;
    {
      std::unique_lock<std::mutex> lk(m_in_mu);
      m_in_cv.wait(lk, [this] { return m_stop || m_pending.has_value(); });
      if (m_stop && !m_pending.has_value())
        return;
      frame = std::move(*m_pending);
      m_pending.reset();
    }

    // Honor reset flag from emu thread before assembling features.
    if (frame.reset_context)
    {
      m_kv_cache.fill(0.0f);
      m_prev_labels.fill(0.0f);
    }

    if (static_cast<int>(frame.core_features.size()) != AIModelDims::CORE_FEATURE_DIM)
    {
      WARN_LOG_FMT(CORE, "LocalOnnxBackend: dropping frame with bad core size {}",
                   frame.core_features.size());
      continue;
    }

    std::vector<float> features = BuildFullFeatures(frame.core_features);

    // Sample a feature row every ~2s for behavioral debugging.
    if (++feat_log_counter % 120 == 1)
    {
      const int N = AIModelDims::FEATURE_DIM;
      INFO_LOG_FMT(CORE,
                   "AIController feat[0..10]: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} "
                   "{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}",
                   features[0], features[1], features[2], features[3],
                   features[4], features[5], features[6], features[7],
                   features[8], features[9], features[10]);
      INFO_LOG_FMT(CORE,
                   "AIController feat[183..193] (prev_labels): "
                   "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} "
                   "{:.3f} {:.3f} {:.3f} {:.3f}",
                   features[N-11], features[N-10], features[N-9], features[N-8],
                   features[N-7], features[N-6], features[N-5],
                   features[N-4], features[N-3], features[N-2], features[N-1]);
    }

    try
    {
      Ort::MemoryInfo mem_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

      int64_t feat_shape[] = {1, AIModelDims::FEATURE_DIM};
      Ort::Value feat_tensor = Ort::Value::CreateTensor<float>(
          mem_info, features.data(), AIModelDims::FEATURE_DIM, feat_shape, 2);

      int64_t kv_shape[] = {AIModelDims::KV_CACHE_LAYERS, 2, 1,
                            AIModelDims::KV_CACHE_SEQ, AIModelDims::KV_CACHE_DIM};
      Ort::Value kv_tensor = Ort::Value::CreateTensor<float>(
          mem_info, m_kv_cache.data(), AIModelDims::KV_CACHE_SIZE, kv_shape, 5);

      std::array<Ort::Value, 2> inputs = {std::move(feat_tensor), std::move(kv_tensor)};

      const char* input_names[]  = {"features", "kv_cache_in"};
      const char* output_names[] = {"btn_probs", "stick_vals", "kv_cache_out"};

      auto t0 = std::chrono::steady_clock::now();
      auto outputs = m_session->Run(Ort::RunOptions{nullptr},
                                    input_names, inputs.data(), 2,
                                    output_names, 3);
      auto t1 = std::chrono::steady_clock::now();

      float ms_infer = std::chrono::duration<float, std::milli>(t1 - t0).count();
      stats_infer.Push(ms_infer);
      if (stats_infer.Full())
      {
        stats_infer.Dump("infer_ms   ");
        stats_infer.Reset();
      }

      const float* btn_probs  = outputs[0].GetTensorData<float>();
      const float* stick_vals = outputs[1].GetTensorData<float>();

      if (++infer_log_counter % 120 == 1)
      {
        INFO_LOG_FMT(CORE,
                     "AIController raw: btn=[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}] "
                     "stk=[{:.3f},{:.3f},{:.3f},{:.3f}]",
                     btn_probs[0], btn_probs[1], btn_probs[2], btn_probs[3],
                     btn_probs[4], btn_probs[5], btn_probs[6],
                     stick_vals[0], stick_vals[1], stick_vals[2], stick_vals[3]);
      }

      GCPadStatus pad = DecodeOutput(btn_probs, stick_vals, frame.mirror_x);

      // Per-frame log of the pad delivered to the game.
      INFO_LOG_FMT(CORE,
                   "AI pad: {}{}{}{}{}{}{}  stk=({},{}) cstic=({},{})  raw=[{:.2f},{:.2f},{:.2f},"
                   "{:.2f},{:.2f},{:.2f},{:.2f}]",
                   (pad.button & PAD_BUTTON_A)     ? "A" : ".",
                   (pad.button & PAD_BUTTON_B)     ? "B" : ".",
                   (pad.button & PAD_BUTTON_X)     ? "X" : ".",
                   (pad.button & PAD_BUTTON_Y)     ? "Y" : ".",
                   (pad.button & PAD_TRIGGER_L)    ? "L" : ".",
                   (pad.button & PAD_TRIGGER_R)    ? "R" : ".",
                   (pad.button & PAD_BUTTON_START) ? "S" : ".",
                   pad.stickX, pad.stickY, pad.substickX, pad.substickY,
                   btn_probs[0], btn_probs[1], btn_probs[2], btn_probs[3],
                   btn_probs[4], btn_probs[5], btn_probs[6]);

      // Carry KV cache forward.
      const float* kv_out = outputs[2].GetTensorData<float>();
      std::copy(kv_out, kv_out + AIModelDims::KV_CACHE_SIZE, m_kv_cache.begin());

      // Update prev_labels from the actual pad we just sent.  Mirrors the
      // pre-worker logic exactly so the next frame's input matches training.
      m_prev_labels[0] = (pad.button & PAD_BUTTON_A)  ? 1.0f : 0.0f;
      m_prev_labels[1] = (pad.button & PAD_BUTTON_B)  ? 1.0f : 0.0f;
      m_prev_labels[2] = (pad.button & PAD_BUTTON_X)  ? 1.0f : 0.0f;
      m_prev_labels[3] = (pad.button & PAD_BUTTON_Y)  ? 1.0f : 0.0f;
      m_prev_labels[4] = (pad.button & PAD_TRIGGER_L) && (pad.button & PAD_BUTTON_A) ? 1.0f : 0.0f;
      m_prev_labels[5] = (pad.button & PAD_TRIGGER_L) && (pad.button & PAD_BUTTON_B) ? 1.0f : 0.0f;
      m_prev_labels[6] = (pad.button & PAD_TRIGGER_R) ? 1.0f : 0.0f;
      for (int i = 0; i < AIModelDims::STICK_DIM_OUT; i++)
        m_prev_labels[AIModelDims::BUTTON_DIM_OUT + i] = stick_vals[i];

      // Publish to output slot.
      {
        std::lock_guard<std::mutex> lk(m_out_mu);
        m_last_output = pad;
        m_has_output  = true;
      }
    }
    catch (const Ort::Exception& e)
    {
      WARN_LOG_FMT(CORE, "LocalOnnxBackend: ORT inference error: {}", e.what());
      std::lock_guard<std::mutex> lk(m_out_mu);
      m_has_output = false;
    }
  }
}

GCPadStatus LocalOnnxBackend::DecodeOutput(const float* btn_probs, const float* stick_vals,
                                            bool mirror_x)
{
  GCPadStatus pad{};
  pad.isConnected = true;
  pad.button      = PAD_USE_ORIGIN;

  bool btn_state[AIModelDims::BUTTON_DIM_OUT];
  for (int i = 0; i < AIModelDims::BUTTON_DIM_OUT; i++)
    btn_state[i] = btn_probs[i] > 0.5f;

  bool has_a         = btn_state[0];
  bool has_b         = btn_state[1];
  bool has_x         = btn_state[2];
  bool has_y         = btn_state[3];
  bool has_lob_pass  = btn_state[4];
  bool has_chip_shot = btn_state[5];
  bool has_r         = btn_state[6];

  if (has_a || has_lob_pass)
  {
    pad.button |= PAD_BUTTON_A;
    pad.analogA = 0xFF;
  }
  if (has_b || has_chip_shot)
  {
    pad.button |= PAD_BUTTON_B;
    pad.analogB = 0xFF;
  }
  if (has_x)  pad.button |= PAD_BUTTON_X;
  if (has_y)  pad.button |= PAD_BUTTON_Y;
  if (has_lob_pass || has_chip_shot)
  {
    pad.button |= PAD_TRIGGER_L;
    pad.triggerLeft = 0xFF;
  }
  if (has_r)
  {
    pad.button |= PAD_TRIGGER_R;
    pad.triggerRight = 0xFF;
  }

  auto DecodeStick = [](float val) -> uint8_t {
    float raw = val * 128.0f + 128.0f;
    return static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, raw)));
  };

  float stick_x  = stick_vals[0];
  float stick_y  = stick_vals[1];
  float cstick_x = stick_vals[2];
  float cstick_y = stick_vals[3];

  // C-stick deadzone: snap small values to neutral so only intentional dekes register.
  constexpr float kCStickDeadzone = 0.4f;
  if (std::abs(cstick_x) < kCStickDeadzone) cstick_x = 0.0f;
  if (std::abs(cstick_y) < kCStickDeadzone) cstick_y = 0.0f;

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
}  // namespace

// ---------------------------------------------------------------------------
// AIController — emu-thread façade.  Reads game state, submits to backend.
// ---------------------------------------------------------------------------

AIController::AIController() = default;
AIController::~AIController() { Shutdown(); }

bool AIController::Load(const std::string& onnx_path)
{
  Shutdown();
  auto backend = std::make_unique<LocalOnnxBackend>();
  if (!backend->Load(onnx_path))
    return false;
  m_backend = std::move(backend);
  return true;
}

void AIController::Shutdown()
{
  if (m_backend)
  {
    m_backend->Shutdown();
    m_backend.reset();
  }
  m_prev_phase_family = -1;
  m_phase_active      = false;
}

bool AIController::IsLoaded() const { return m_backend != nullptr; }

bool AIController::IsMatchActive() const
{
  return m_phase_active && m_backend && m_backend->HasOutput();
}

GCPadStatus AIController::GetLastOutput() const
{
  if (!m_backend)
  {
    GCPadStatus pad{};
    pad.stickX = pad.stickY = pad.substickX = pad.substickY = 0x80;
    pad.isConnected = true;
    return pad;
  }
  return m_backend->GetLastOutput();
}

void AIController::OnFrameEnd(int controlled_port, bool mirror_x)
{
  // Emu-thread windowed timing buckets.  infer_ms now lives on the worker
  // thread (see LocalOnnxBackend::WorkerLoop); here we track gs_ms (cost of
  // ReadGameStateCore) and interval_ms (true emu-thread cadence).
  static WindowedStats s_stats_gs(10.0f);
  static WindowedStats s_stats_interval(17.0f);  // >16.67ms = below 60fps
  static std::chrono::steady_clock::time_point s_last_frame_end{};
  auto invalidate_interval = []() {
    s_last_frame_end = std::chrono::steady_clock::time_point{};
  };

  if (!m_backend)
  {
    invalidate_interval();
    return;
  }

  // Read game phase to detect segment boundaries (mirrors build_dataset.py segmentation).
  constexpr uint32_t CGAME_SINGLETON = 0x80373708;
  uint32_t cGamePtr   = Memory::Read_U32(CGAME_SINGLETON);
  uint32_t game_phase = (cGamePtr != 0) ? Memory::Read_U32(cGamePtr + 0x24) : 0;

  // phase_family: 1=kickoff, 4=active play (phases 4 and 5), -1=everything else
  int phase_family = -1;
  if (game_phase == 1)
    phase_family = 1;
  else if (game_phase == 4 || game_phase == 5)
    phase_family = 4;

  bool phase_changed  = (phase_family != m_prev_phase_family);
  m_prev_phase_family = phase_family;

  if (phase_family == -1)
  {
    // Outside active play (celebrations, menus, etc.) — backend will reset
    // its KV/prev_labels on the next active frame via reset_context.
    m_phase_active = false;
    invalidate_interval();
    return;
  }

  m_phase_active = true;

  auto t_frame_start = std::chrono::steady_clock::now();

  if (s_last_frame_end != std::chrono::steady_clock::time_point{})
  {
    float interval_ms =
        std::chrono::duration<float, std::milli>(t_frame_start - s_last_frame_end).count();
    s_stats_interval.Push(interval_ms);
  }

  std::vector<float> core_features = ReadGameStateCore(controlled_port, mirror_x);
  if (core_features.empty())
  {
    invalidate_interval();
    return;
  }

  auto t_gs_end = std::chrono::steady_clock::now();
  float ms_gs   = std::chrono::duration<float, std::milli>(t_gs_end - t_frame_start).count();
  s_stats_gs.Push(ms_gs);
  s_last_frame_end = t_gs_end;

  // Hand off to the worker — non-blocking.
  AIInputFrame frame;
  frame.core_features = std::move(core_features);
  frame.reset_context = phase_changed;
  frame.mirror_x      = mirror_x;
  m_backend->Submit(std::move(frame));

  // Dump emu-thread stats every ~10s.
  if (s_stats_gs.Full())
  {
    s_stats_gs.Dump      ("gs_ms      ");
    s_stats_interval.Dump("interval_ms");
    s_stats_gs.Reset();
    s_stats_interval.Reset();
  }
}

// ---------------------------------------------------------------------------
// Feature extraction — mirrors build_dataset.py::extract_features() v6 exactly
// ---------------------------------------------------------------------------
// Returns the 183-float CORE feature vector. The trailing 11 prev_labels are
// appended by BuildFullFeatures() to form the 194-float model input.
// Full layout (194 total = 183 core + 11 prev_labels):
//   1. Ball (11): pos(3) + vel(3) + charge(1) + perfect_pass(1) + ball_to_goal(3)
//   2. Ball owner one-hot (11)
//   3. Self character (19): Δpos(3) + state_idx(1) + heading(2) + goal(3)
//      + effect_oh(5) + speed_item_oh(3) + item_timer(1) + is_carrier(1)
//   4. 3 friendly strikers (7 × 3 = 21): Δpos(3) + state_idx(1) + heading(2)
//      + is_carrier(1)
//   5. Friendly goalie (4): Δpos(3) + state_idx(1)
//   6. 4 enemy strikers (7 × 4 = 28): Δpos(3) + state_idx(1) + heading(2)
//      + is_carrier(1)
//   7. Enemy goalie (4): Δpos(3) + state_idx(1)
//   8. Own inventory (2 × 11 = 22)
//   9. Enemy inventory (2 × 11 = 22)
//  10. 4 nearest field items (8 × 4 = 32): type_idx(1) + Δpos(3) + vel(3)
//      + strength(1)
//  11. Tactical summary (5)
//  12. Possession booleans (2)
//  13. Phase booleans (2): is_kickoff + goalie_has_ball
//  14. Previous frame action (11): 7 buttons + 4 sticks

std::vector<float> AIController::ReadGameStateCore(int controlled_port, bool mirror_x) const
{
  const AddressSpace::Accessors* acc =
      AddressSpace::GetAccessors(AddressSpace::Type::Effective);

  const float sign = mirror_x ? -1.0f : 1.0f;
  auto mx = [sign](float x) { return sign * x; };

  // Read character pointers
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

  // mirror_x=false → LEFT team (slots 0-3, goalie 4)
  // mirror_x=true  → RIGHT team (slots 5-8, goalie 9)
  const int* own_striker_slots   = mirror_x ? RIGHT_STRIKER_SLOTS : LEFT_STRIKER_SLOTS;
  const int  own_goalie_slot     = mirror_x ? RIGHT_GOALIE_SLOT   : LEFT_GOALIE_SLOT;
  const int* enemy_striker_slots = mirror_x ? LEFT_STRIKER_SLOTS  : RIGHT_STRIKER_SLOTS;
  const int  enemy_goalie_slot   = mirror_x ? LEFT_GOALIE_SLOT    : RIGHT_GOALIE_SLOT;

  // Find user-controlled slot on our team
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

  uint32_t ball_ptr        = Memory::Read_U32(Metadata::addressBallPointer);
  float    bvx = 0.0f, bvy = 0.0f, bvz = 0.0f;
  uint32_t ball_owner_ptr  = 0;
  bool     is_perfect_pass = false;
  if (ball_ptr != 0)
  {
    bvx             = mx(acc->ReadF32(ball_ptr + 0x58));
    bvy             =    acc->ReadF32(ball_ptr + 0x5C);
    bvz             =    acc->ReadF32(ball_ptr + 0x60);
    ball_owner_ptr  = Memory::Read_U32(ball_ptr + 0x24);
    is_perfect_pass = (Memory::Read_U8(ball_ptr + 0xA1) != 0);
  }

  float ball_charge = static_cast<float>(Memory::Read_U32(Metadata::addressChargedBallAmount));

  // Resolve ball owner slot (0-9, or 10 = no owner)
  int owner_slot = 10;
  for (int i = 0; i < 10; i++)
  {
    if (char_ptrs[i] != 0 && char_ptrs[i] == ball_owner_ptr)
    {
      owner_slot = i;
      break;
    }
  }

  float goal_x = std::abs(acc->ReadF32(0x802a3e60));

  uint32_t left_team_ptr  = Memory::Read_U32(Metadata::addressTeam1Pointer);
  uint32_t right_team_ptr = Memory::Read_U32(Metadata::addressTeam2Pointer);
  uint32_t own_team_ptr   = mirror_x ? right_team_ptr : left_team_ptr;
  uint32_t enemy_team_ptr = mirror_x ? left_team_ptr  : right_team_ptr;

  // Read game phase for is_kickoff feature
  constexpr uint32_t CGAME_SINGLETON = 0x80373708;
  uint32_t cGamePtr   = Memory::Read_U32(CGAME_SINGLETON);
  uint32_t game_phase = (cGamePtr != 0) ? Memory::Read_U32(cGamePtr + 0x24) : 0;

  // Self position (needed for field items + tactical)
  float sc_px = mx(chars[self_slot].pos_x);
  float sc_py = chars[self_slot].pos_y;

  // Debug: log key game state reads every 120 frames
  static int s_dbg_counter = 0;
  if (++s_dbg_counter % 120 == 1)
  {
    INFO_LOG_FMT(CORE, "AIController GS: ball_ptr={:#010x} bpx={:.2f} bpy={:.2f} "
                 "self_slot={} self_pos=({:.2f},{:.2f}) phase={}",
                 ball_ptr, bpx, bpy, self_slot, sc_px, sc_py, game_phase);
  }

  // ── Build core feature vector (183 dims; prev_labels appended later) ───────
  std::vector<float> feat;
  feat.reserve(AIModelDims::CORE_FEATURE_DIM);

  // --- 1. Ball (11) ---
  feat.push_back(bpx);
  feat.push_back(bpy);
  feat.push_back(bpz);
  feat.push_back(bvx);
  feat.push_back(bvy);
  feat.push_back(bvz);
  feat.push_back(ball_charge / 35.0f);
  feat.push_back(is_perfect_pass ? 1.0f : 0.0f);
  // Ball-to-goal geometry (v6)
  {
    float b2g_dx   = goal_x - bpx;
    float b2g_dy   = 0.0f - bpy;
    float b2g_dist = std::sqrt(b2g_dx * b2g_dx + b2g_dy * b2g_dy) + 1e-6f;
    feat.push_back(b2g_dist / 40.0f);
    feat.push_back(b2g_dx / b2g_dist);
    feat.push_back(b2g_dy / b2g_dist);
  }

  // --- 2. Ball owner one-hot (11) ---
  AppendOneHot(feat, owner_slot, 11);

  // --- 3. Self character (19) ---
  {
    const CharData& sc = chars[self_slot];
    float dx   = goal_x - sc_px;
    float dy   = sc_py;
    float dist = std::sqrt(dx * dx + dy * dy) + 1e-6f;

    auto [sin_h, cos_h] = HeadingSinCos(sc.heading);
    if (mirror_x)
      cos_h = -cos_h;

    feat.push_back(sc_px - bpx);                         // Δpos to ball (3)
    feat.push_back(sc_py - bpy);
    feat.push_back(sc.pos_z);
    feat.push_back(StrikerStateIdx(sc.action_state));     // state index (1)
    feat.push_back(sin_h);                                // heading (2)
    feat.push_back(cos_h);
    feat.push_back(dist / 40.0f);                         // goal dist+angle (3)
    feat.push_back(dx / dist);
    feat.push_back(dy / dist);
    AppendEffectOH(feat, sc.effect_type);                 // effect (5)
    AppendSpeedItemOH(feat, sc.speed_item_type);          // speed item (3)
    feat.push_back(sc.speed_item_timer / 10.0f);          // item timer (1)
    feat.push_back(self_slot == owner_slot ? 1.0f : 0.0f);  // is ball carrier (1)
  }

  // --- 4. 3 other friendly strikers sorted by distance to ball (3 × 7 = 21) ---
  {
    struct SlotDist { int slot; float dist2; };
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
        feat.push_back(ch_px - bpx);                      // Δpos (3)
        feat.push_back(ch.pos_y - bpy);
        feat.push_back(ch.pos_z);
        feat.push_back(StrikerStateIdx(ch.action_state));  // state index (1)
        feat.push_back(sh);                                // heading (2)
        feat.push_back(ch_cos);
        feat.push_back(s == owner_slot ? 1.0f : 0.0f);    // is carrier (1)
      }
      else
      {
        for (int z = 0; z < 7; z++)
          feat.push_back(0.0f);
      }
    }
  }

  // --- 5. Friendly goalie (4) ---
  {
    const CharData& gs = chars[own_goalie_slot];
    feat.push_back(mx(gs.pos_x) - bpx);                   // Δpos (3)
    feat.push_back(gs.pos_y - bpy);
    feat.push_back(gs.pos_z);
    feat.push_back(GoalieStateIdx(gs.action_state));       // state index (1)
  }

  // --- 6. 4 enemy strikers sorted by distance to ball (4 × 7 = 28) ---
  {
    struct SlotDist { int slot; float dist2; };
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
      feat.push_back(ch_px - bpx);                        // Δpos (3)
      feat.push_back(ch.pos_y - bpy);
      feat.push_back(ch.pos_z);
      feat.push_back(StrikerStateIdx(ch.action_state));    // state index (1)
      feat.push_back(sh);                                  // heading (2)
      feat.push_back(ch_cos);
      feat.push_back(s == owner_slot ? 1.0f : 0.0f);      // is carrier (1)
    }
  }

  // --- 7. Enemy goalie (4) ---
  {
    const CharData& eg = chars[enemy_goalie_slot];
    feat.push_back(mx(eg.pos_x) - bpx);                   // Δpos (3)
    feat.push_back(eg.pos_y - bpy);
    feat.push_back(eg.pos_z);
    feat.push_back(GoalieStateIdx(eg.action_state));       // state index (1)
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

  // --- 10. 4 nearest field items (8 × 4 = 32) ---
  {
    std::vector<FieldItem> all_items = ReadFieldItems(acc);

    struct ItemDist { float dist2; int idx; };
    std::vector<ItemDist> sorted_items;
    sorted_items.reserve(all_items.size());
    for (int i = 0; i < static_cast<int>(all_items.size()); i++)
    {
      float ipx = mx(all_items[i].pos_x);
      float ipy = all_items[i].pos_y;
      float dx2 = ipx - sc_px;
      float dy2 = ipy - sc_py;
      sorted_items.push_back({dx2 * dx2 + dy2 * dy2, i});
    }
    std::sort(sorted_items.begin(), sorted_items.end(),
              [](const ItemDist& a, const ItemDist& b) { return a.dist2 < b.dist2; });

    for (int k = 0; k < FIELD_ITEM_K; k++)
    {
      if (k < static_cast<int>(sorted_items.size()))
      {
        const FieldItem& fi = all_items[sorted_items[k].idx];
        float ipx = mx(fi.pos_x);
        float ipy = fi.pos_y;
        feat.push_back(static_cast<float>(std::min(fi.powerup_type, 8)));  // type idx (1)
        feat.push_back(ipx - sc_px);            // Δpos to self (3)
        feat.push_back(ipy - sc_py);
        feat.push_back(fi.pos_z);               // pos_z is height, not mirrored
        feat.push_back(mx(fi.vel_x));           // velocity (3)
        feat.push_back(fi.vel_y);
        feat.push_back(fi.vel_z);
        feat.push_back(fi.strength_level / 2.0f);  // strength (1)
      }
      else
      {
        feat.push_back(9.0f);  // padding type index (empty slot)
        for (int z = 0; z < 7; z++)
          feat.push_back(0.0f);
      }
    }
  }

  // --- 11. Tactical summary (5) ---
  {
    float sdx       = sc_px - bpx;
    float sdy       = sc_py - bpy;
    float self_dist = std::sqrt(sdx * sdx + sdy * sdy);

    int self_rank = 0;
    for (int i = 0; i < 4; i++)
    {
      int s = own_striker_slots[i];
      if (s == self_slot)
        continue;
      float dx2 = mx(chars[s].pos_x) - bpx;
      float dy2 = chars[s].pos_y - bpy;
      if (dx2 * dx2 + dy2 * dy2 < self_dist * self_dist)
        self_rank++;
    }
    feat.push_back(self_dist / 40.0f);
    feat.push_back(static_cast<float>(self_rank) / 3.0f);
    feat.push_back(self_rank == 0 ? 1.0f : 0.0f);

    // Nearest enemy to self
    float min_en_dist2 = 1e18f;
    int   nearest_en   = enemy_striker_slots[0];
    for (int i = 0; i < 4; i++)
    {
      int   s  = enemy_striker_slots[i];
      float ex = mx(chars[s].pos_x) - sc_px;
      float ey = chars[s].pos_y - sc_py;
      float d2 = ex * ex + ey * ey;
      if (d2 < min_en_dist2)
      {
        min_en_dist2 = d2;
        nearest_en   = s;
      }
    }
    float nearest_enemy_dist = std::sqrt(min_en_dist2);
    feat.push_back(nearest_enemy_dist / 40.0f);

    float goal_dx  = goal_x - sc_px;
    float goal_dy  = 0.0f - sc_py;
    float goal_len = std::sqrt(goal_dx * goal_dx + goal_dy * goal_dy) + 1e-6f;
    float en_dx    = mx(chars[nearest_en].pos_x) - sc_px;
    float en_dy    = chars[nearest_en].pos_y - sc_py;
    float en_len   = nearest_enemy_dist + 1e-6f;
    feat.push_back((goal_dx / goal_len) * (en_dx / en_len) +
                   (goal_dy / goal_len) * (en_dy / en_len));
  }

  // --- 12. Possession booleans (2) ---
  {
    bool friendly_has_ball = false;
    if (owner_slot != 10)
    {
      for (int i = 0; i < 4; i++)
      {
        if (own_striker_slots[i] == owner_slot)
        {
          friendly_has_ball = true;
          break;
        }
      }
      if (own_goalie_slot == owner_slot)
        friendly_has_ball = true;
    }
    bool enemy_has_ball = (owner_slot != 10 && !friendly_has_ball);
    feat.push_back(friendly_has_ball ? 1.0f : 0.0f);
    feat.push_back(enemy_has_ball    ? 1.0f : 0.0f);
  }

  // --- 13. Phase booleans (2) ---
  feat.push_back(game_phase == 1 ? 1.0f : 0.0f);          // is_kickoff
  feat.push_back(owner_slot == own_goalie_slot ? 1.0f : 0.0f);  // goalie_has_ball

  // --- 14. Previous frame action (11) is appended by BuildFullFeatures(). ----

  if (static_cast<int>(feat.size()) != AIModelDims::CORE_FEATURE_DIM)
  {
    ERROR_LOG_FMT(CORE, "AIController: core feature dim mismatch: got {} expected {}",
                  feat.size(), AIModelDims::CORE_FEATURE_DIM);
    return {};
  }

  return feat;
}

}  // namespace Movie

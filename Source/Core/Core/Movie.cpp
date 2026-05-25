// Copyright 2009 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Core/Movie.h"

#include <filesystem>  // C++17
#include <fstream>
#include <iostream>
namespace fs = std::filesystem;
#include "unzip.h"
#include <picojson.h>
#include <zstd.h>

// Enable LZMA2 decompressor plugin for old-format diffs (created by hdiffz -c-lzma2).
#define _CompressPlugin_lzma2
#define _IsNeedIncludeDefaultCompressHead 1
// CommonTypes.h defines LONG as a macro on non-Windows; undef it before
// lzma's 7zTypes.h which uses LONG as a typedef name.
#undef LONG
#include "decompress_plugin_demo.h"

#include "libHDiffPatch/HPatch/patch.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <mbedtls/config.h>
#include <mbedtls/md.h>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include "Common/Assert.h"
#include "Common/ChunkFile.h"
#include "Common/CommonPaths.h"
#include "Common/Config/Config.h"
#include "Common/FileUtil.h"
#include "Common/Hash.h"
#include "Common/IOFile.h"
#include "Common/MsgHandler.h"
#include "Common/NandPaths.h"
#include "Common/StringUtil.h"
#include "Common/Timer.h"
#include "Common/Version.h"

#include "Core/Boot/Boot.h"
#include "Core/Config/MainSettings.h"
#include "Core/Config/SYSCONFSettings.h"
#include "Core/Config/WiimoteSettings.h"
#include "Core/ConfigLoaders/MovieConfigLoader.h"
#include "Core/ConfigManager.h"
#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/AIController.h"
#include "Core/GameStateFrame.h"
#include "Core/DSP/DSPCore.h"
#include "Core/HW/CPU.h"
#include "Core/HW/DVD/DVDInterface.h"
#include "Core/HW/EXI/EXI.h"
#include "Core/HW/EXI/EXI_DeviceIPL.h"
#include "Core/HW/EXI/EXI_DeviceMemoryCard.h"
#include "Core/HW/ProcessorInterface.h"
#include "Core/HW/SI/SI.h"
#include "Core/HW/SI/SI_Device.h"
#include "Core/HW/Wiimote.h"
#include "Core/HW/WiimoteCommon/DataReport.h"
#include "Core/HW/WiimoteCommon/WiimoteReport.h"

#include "Core/HW/WiimoteEmu/Encryption.h"
#include "Core/HW/WiimoteEmu/Extension/Classic.h"
#include "Core/HW/WiimoteEmu/Extension/Nunchuk.h"
#include "Core/HW/WiimoteEmu/ExtensionPort.h"

#include "Core/IOS/USB/Bluetooth/BTEmu.h"
#include "Core/IOS/USB/Bluetooth/WiimoteDevice.h"
#include "Core/NetPlayProto.h"
#include "Core/State.h"
#include "Core/WiiUtils.h"

#include "DiscIO/Enums.h"

#include "InputCommon/GCPadStatus.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include "VideoCommon/VideoBackendBase.h"
#include "VideoCommon/VideoConfig.h"
#include "Core/StateAuxillary.h"
#include <Core/Metadata.h>

// The chunk to allocate movie data in multiples of.
#define DTM_BASE_LENGTH (1024)

namespace Movie
{
using namespace WiimoteCommon;
using namespace WiimoteEmu;

enum class PlayMode
{
  None = 0,
  Recording,
  Playing,
};

static bool s_bReadOnly = true;
static u32 s_rerecords = 0;
static PlayMode s_playMode = PlayMode::None;

static std::array<ControllerType, 4> s_controllers{};
static std::array<bool, 4> s_wiimotes{};
static ControllerState s_padState;
static std::array<GCPadStatus, 4> s_last_pad_status{};  // Cache of last played/recorded pad status
static DTMHeader tmpHeader;
static std::vector<u8> s_temp_input;
static u64 s_currentByte = 0;
static u64 s_currentFrame = 0, s_totalFrames = 0;  // VI
static u64 s_currentLagCount = 0;
static u64 s_totalLagCount = 0;                               // just stats
static u64 s_currentInputCount = 0, s_totalInputCount = 0;    // just stats
static u64 s_totalTickCount = 0, s_tickCountAtLastInput = 0;  // just stats
static u64 s_recordingStartTime;  // seconds since 1970 that recording started
static bool s_bSaveConfig = false, s_bNetPlay = false;
static bool s_bClearSave = false;
static bool s_bDiscChange = false;
static bool s_bReset = false;
static std::string s_author;
static std::string s_discChange;
static std::array<u8, 16> s_MD5;
static u8 s_bongos, s_memcards;
static std::array<u8, 20> s_revision;
static u32 s_DSPiromHash = 0;
static u32 s_DSPcoefHash = 0;

static bool s_bRecordingFromSaveState = false;
static bool s_bPolled = false;

// CITF playback support - use frame-level inputs from .citframes file instead of DTM
struct CITFFrameInputs
{
  u32 movieFrameNumber;                     // DTM input index this frame corresponds to
  std::array<GCPadStatus, 4> controllers;   // All 4 controller inputs for this frame
};
static std::vector<CITFFrameInputs> s_citf_inputs;  // CITF frames (indexed by capture order, not movieFrameNumber)
static bool s_use_citf_inputs = false;               // True if playing back from CITF instead of DTM
static std::string s_citf_file_path;                 // Path to .citframes file if found
static std::string s_cit_stem_name;                  // Stem of the .cit file being played (e.g. "game1")
static std::string s_cit_dir_path;                   // Parent directory of the .cit file being played

// AI controller — optional ONNX-driven input source.
// When loaded, OnFrameEnd() is called each frame and its output is fed through
// PlayController() in place of DTM/CITF data.
static std::unique_ptr<AIController> s_ai_controller;
static bool s_use_ai_inputs       = false;
static int  s_ai_controlled_port  = 0;
static bool s_ai_mirror_x         = false;

// s_InputDisplay is used by both CPU and GPU (is mutable).
static std::mutex s_input_display_lock;
static std::string s_InputDisplay[8];

static GCManipFunction s_gc_manip_func;
static WiiManipFunction s_wii_manip_func;

static std::string s_current_file_name;

static void GetSettings();
static bool IsMovieHeader(const std::array<u8, 4>& magic)
{
  return magic[0] == 'D' && magic[1] == 'T' && magic[2] == 'M' && magic[3] == 0x1A;
}

static std::array<u8, 20> ConvertGitRevisionToBytes(const std::string& revision)
{
  std::array<u8, 20> revision_bytes{};

  if (revision.size() % 2 == 0 && std::all_of(revision.begin(), revision.end(), ::isxdigit))
  {
    // The revision string normally contains a git commit hash,
    // which is 40 hexadecimal digits long. In DTM files, each pair of
    // hexadecimal digits is stored as one byte, for a total of 20 bytes.
    size_t bytes_to_write = std::min(revision.size() / 2, revision_bytes.size());
    unsigned int temp;
    for (size_t i = 0; i < bytes_to_write; ++i)
    {
      sscanf(&revision[2 * i], "%02x", &temp);
      revision_bytes[i] = temp;
    }
  }
  else
  {
    // If the revision string for some reason doesn't only contain hexadecimal digit
    // pairs, we instead copy the string with no conversion. This probably doesn't match
    // the intended design of the DTM format, but it's the most sensible fallback.
    size_t bytes_to_write = std::min(revision.size(), revision_bytes.size());
    std::copy_n(std::begin(revision), bytes_to_write, std::begin(revision_bytes));
  }

  return revision_bytes;
}

// NOTE: GPU Thread
std::string GetInputDisplay()
{
  if (!IsMovieActive())
  {
    s_controllers = {};
    s_wiimotes = {};
    for (int i = 0; i < 4; ++i)
    {
      if (SerialInterface::GetDeviceType(i) == SerialInterface::SIDEVICE_GC_GBA_EMULATED)
        s_controllers[i] = ControllerType::GBA;
      else if (SerialInterface::GetDeviceType(i) != SerialInterface::SIDEVICE_NONE)
        s_controllers[i] = ControllerType::GC;
      else
        s_controllers[i] = ControllerType::None;
      s_wiimotes[i] = Config::Get(Config::GetInfoForWiimoteSource(i)) != WiimoteSource::None;
    }
  }

  std::string input_display;
  {
    std::lock_guard guard(s_input_display_lock);
    for (int i = 0; i < 4; ++i)
    {
      if (IsUsingPad(i))
        input_display += s_InputDisplay[i] + '\n';
    }
    for (int i = 0; i < 4; ++i)
    {
      if (IsUsingWiimote(i))
        input_display += s_InputDisplay[i + 4] + '\n';
    }
  }
  return input_display;
}

// NOTE: GPU Thread
std::string GetRTCDisplay()
{
  using ExpansionInterface::CEXIIPL;

  const time_t current_time = CEXIIPL::GetEmulatedTime(CEXIIPL::UNIX_EPOCH);
  const tm* const gm_time = gmtime(&current_time);

  std::ostringstream format_time;
  format_time << std::put_time(gm_time, "Date/Time: %c\n");
  return format_time.str();
}

// NOTE: GPU Thread
std::string GetRerecords()
{
  if (IsMovieActive())
    return fmt::format("Rerecords: {}", s_rerecords);

  return "Rerecords: N/A";
}

void FrameUpdate()
{
  s_currentFrame++;
  if (!s_bPolled)
    s_currentLagCount++;

  if (IsRecordingInput())
  {
    s_totalFrames = s_currentFrame;
    s_totalLagCount = s_currentLagCount;
  }

  s_bPolled = false;
}

static void CheckMD5();
static void GetMD5();

// called when game is booting up, even if no movie is active,
// but potentially after BeginRecordingInput or PlayInput has been called.
// NOTE: EmuThread
void Init(const BootParameters& boot)
{
  if (std::holds_alternative<BootParameters::Disc>(boot.parameters))
    s_current_file_name = std::get<BootParameters::Disc>(boot.parameters).path;
  else
    s_current_file_name.clear();

  s_bPolled = false;
  s_bSaveConfig = false;
  if (IsPlayingInput())
  {
    ReadHeader();
    std::thread md5thread(CheckMD5);
    md5thread.detach();
    if (strncmp(tmpHeader.gameID.data(), SConfig::GetInstance().GetGameID().c_str(), 6))
    {
      PanicAlertFmtT("The recorded game ({0}) is not the same as the selected game ({1})",
                     tmpHeader.GetGameID(), SConfig::GetInstance().GetGameID());
      EndPlayInput(false);
    }
  }

  if (IsRecordingInput())
  {
    GetSettings();
    std::thread md5thread(GetMD5);
    md5thread.detach();
    s_tickCountAtLastInput = 0;
  }

  memset(&s_padState, 0, sizeof(s_padState));

  for (auto& disp : s_InputDisplay)
    disp.clear();

  if (!IsMovieActive())
  {
    s_bRecordingFromSaveState = false;
    s_rerecords = 0;
    s_currentByte = 0;
    s_currentFrame = 0;
    s_currentLagCount = 0;
    s_currentInputCount = 0;
  }

  // AI Controller: initialize from INI at every game boot so the model works
  // without needing a movie file to be playing.  AIIpcPort takes precedence
  // over AIModelPath: a configured IPC port means an external Python trainer
  // owns the policy, and AIModelPath is irrelevant.
  {
    int ai_ipc_port = Config::Get(Config::MAIN_MOVIE_AI_IPC_PORT);
    std::string ai_model_path = Config::Get(Config::MAIN_MOVIE_AI_MODEL_PATH);
    INFO_LOG_FMT(CORE, "Movie::Init — AIIpcPort={} AIModelPath='{}'",
                 ai_ipc_port, ai_model_path);
    int ai_port = Config::Get(Config::MAIN_MOVIE_AI_CONTROLLED_PORT);
    bool ai_mirror_x = Config::Get(Config::MAIN_MOVIE_AI_MIRROR_X);
    if (ai_ipc_port > 0)
      InitAIControllerIpc(ai_ipc_port, ai_port, ai_mirror_x);
    else if (!ai_model_path.empty())
      InitAIController(ai_model_path, ai_port, ai_mirror_x);
  }
}

// NOTE: CPU Thread
void InputUpdate()
{
  s_currentInputCount++;
  if (IsRecordingInput())
  {
    s_totalInputCount = s_currentInputCount;
    s_totalTickCount += CoreTiming::GetTicks() - s_tickCountAtLastInput;
    s_tickCountAtLastInput = CoreTiming::GetTicks();
  }
}

// NOTE: CPU Thread
void SetPolledDevice()
{
  s_bPolled = true;
}

// NOTE: Host Thread
void SetReadOnly(bool bEnabled)
{
  if (s_bReadOnly != bEnabled)
    Core::DisplayMessage(bEnabled ? "Read-only mode." : "Read+Write mode.", 1000);

  s_bReadOnly = bEnabled;
}

bool IsRecordingInput()
{
  return (s_playMode == PlayMode::Recording);
}

bool IsRecordingInputFromSaveState()
{
  return s_bRecordingFromSaveState;
}

bool IsJustStartingRecordingInputFromSaveState()
{
  return IsRecordingInputFromSaveState() && s_currentFrame == 0;
}

bool IsJustStartingPlayingInputFromSaveState()
{
  return IsRecordingInputFromSaveState() && s_currentFrame == 1 && IsPlayingInput();
}

bool IsPlayingInput()
{
  return (s_playMode == PlayMode::Playing);
}

bool IsMovieActive()
{
  return s_playMode != PlayMode::None;
}

bool IsReadOnly()
{
  return s_bReadOnly;
}

u64 GetRecordingStartTime()
{
  return s_recordingStartTime;
}

u64 GetCurrentFrame()
{
  return s_currentFrame;
}

u64 GetTotalFrames()
{
  return s_totalFrames;
}

u64 GetCurrentInputCount()
{
  return s_currentInputCount;
}

std::string GetCITStemName()
{
  return s_cit_stem_name;
}

std::string GetCITDirPath()
{
  return s_cit_dir_path;
}

const GCPadStatus* GetCITFInput(int controller, u64 inputCount)
{
  if (!s_use_citf_inputs || s_citf_inputs.empty())
    return nullptr;

  if (controller < 0 || controller >= 4)
    return nullptr;

  // Find the FIRST CITF frame with matching movieFrameNumber
  // NOTE: We may have DUPLICATE movieFrameNumbers because PadStatus::Update is called
  // twice per frame (fixed + variable timestep). We need to consistently use the FIRST one.
  auto it = std::lower_bound(
      s_citf_inputs.begin(), s_citf_inputs.end(), inputCount,
      [](const CITFFrameInputs& frame, u64 target_input_count) {
        return frame.movieFrameNumber < target_input_count;
      });

  // lower_bound returns the first element >= inputCount
  // Check if we found an exact match
  if (it != s_citf_inputs.end() && it->movieFrameNumber == inputCount)
  {
    // Found a match! lower_bound guarantees this is the FIRST frame with this movieFrameNumber
    return &it->controllers[controller];
  }

  return nullptr;  // No matching frame found
}

u64 GetTotalInputCount()
{
  return s_totalInputCount;
}

u64 GetCurrentLagCount()
{
  return s_currentLagCount;
}

u64 GetTotalLagCount()
{
  return s_totalLagCount;
}

GCPadStatus GetLastPadStatus(int controllerID)
{
  if (controllerID < 0 || controllerID >= 4)
    return GCPadStatus{};
  return s_last_pad_status[controllerID];
}

void SetClearSave(bool enabled)
{
  s_bClearSave = enabled;
}

void SignalDiscChange(const std::string& new_path)
{
  if (Movie::IsRecordingInput())
  {
    size_t size_of_path_without_filename = new_path.find_last_of("/\\") + 1;
    std::string filename = new_path.substr(size_of_path_without_filename);
    constexpr size_t maximum_length = sizeof(DTMHeader::discChange);
    if (filename.length() > maximum_length)
    {
      PanicAlertFmtT("The disc change to \"{0}\" could not be saved in the .dtm file.\n"
                     "The filename of the disc image must not be longer than 40 characters.",
                     filename);
    }
    s_discChange = filename;
    s_bDiscChange = true;
  }
}

void SetReset(bool reset)
{
  s_bReset = reset;
}

bool IsUsingPad(int controller)
{
  return s_controllers[controller] != ControllerType::None;
}

bool IsUsingBongo(int controller)
{
  return ((s_bongos & (1 << controller)) != 0);
}

bool IsUsingGBA(int controller)
{
  return s_controllers[controller] == ControllerType::GBA;
}

bool IsUsingWiimote(int wiimote)
{
  return s_wiimotes[wiimote];
}

bool IsConfigSaved()
{
  return s_bSaveConfig;
}

bool IsStartingFromClearSave()
{
  return s_bClearSave;
}

bool IsUsingMemcard(ExpansionInterface::Slot slot)
{
  switch (slot)
  {
  case ExpansionInterface::Slot::A:
    return (s_memcards & 1) != 0;
  case ExpansionInterface::Slot::B:
    return (s_memcards & 2) != 0;
  default:
    return false;
  }
}

bool IsNetPlayRecording()
{
  return s_bNetPlay;
}

// NOTE: Host Thread
void ChangePads()
{
  if (!Core::IsRunning())
    return;

  ControllerTypeArray controllers{};

  for (int i = 0; i < SerialInterface::MAX_SI_CHANNELS; ++i)
  {
    const SerialInterface::SIDevices si_device = Config::Get(Config::GetInfoForSIDevice(i));
    if (si_device == SerialInterface::SIDEVICE_GC_GBA_EMULATED)
      controllers[i] = ControllerType::GBA;
    else if (SerialInterface::SIDevice_IsGCController(si_device))
      controllers[i] = ControllerType::GC;
    else
      controllers[i] = ControllerType::None;
  }

  if (s_controllers == controllers)
    return;

  for (int i = 0; i < SerialInterface::MAX_SI_CHANNELS; ++i)
  {
    SerialInterface::SIDevices device = SerialInterface::SIDEVICE_NONE;
    if (IsUsingGBA(i))
    {
      device = SerialInterface::SIDEVICE_GC_GBA_EMULATED;
    }
    else if (IsUsingPad(i))
    {
      const SerialInterface::SIDevices si_device = Config::Get(Config::GetInfoForSIDevice(i));
      if (SerialInterface::SIDevice_IsGCController(si_device))
      {
        device = si_device;
      }
      else
      {
        device = IsUsingBongo(i) ? SerialInterface::SIDEVICE_GC_TARUKONGA :
                                   SerialInterface::SIDEVICE_GC_CONTROLLER;
      }
    }

    SerialInterface::ChangeDevice(device, i);
  }
}

// NOTE: Host / Emu Threads
void ChangeWiiPads(bool instantly)
{
  WiimoteEnabledArray wiimotes{};

  for (int i = 0; i < MAX_WIIMOTES; ++i)
  {
    wiimotes[i] = Config::Get(Config::GetInfoForWiimoteSource(i)) != WiimoteSource::None;
  }

  // This is important for Wiimotes, because they can desync easily if they get re-activated
  if (instantly && s_wiimotes == wiimotes)
    return;

  const auto bt = WiiUtils::GetBluetoothEmuDevice();
  for (int i = 0; i < MAX_WIIMOTES; ++i)
  {
    const bool is_using_wiimote = IsUsingWiimote(i);

    Config::SetCurrent(Config::GetInfoForWiimoteSource(i),
                       is_using_wiimote ? WiimoteSource::Emulated : WiimoteSource::None);
    if (bt != nullptr)
      bt->AccessWiimoteByIndex(i)->Activate(is_using_wiimote);
  }
}

// NOTE: Host Thread
bool BeginRecordingInput(const ControllerTypeArray& controllers,
                         const WiimoteEnabledArray& wiimotes)
{
  if (s_playMode != PlayMode::None ||
      (controllers == ControllerTypeArray{} && wiimotes == WiimoteEnabledArray{}))
    return false;

  Core::RunAsCPUThread([controllers, wiimotes] {
    s_controllers = controllers;
    s_wiimotes = wiimotes;
    s_currentFrame = s_totalFrames = 0;
    s_currentLagCount = s_totalLagCount = 0;
    s_currentInputCount = s_totalInputCount = 0;
    s_totalTickCount = s_tickCountAtLastInput = 0;
    s_bongos = 0;
    s_memcards = 0;
    if (NetPlay::IsNetPlayRunning())
    {
      s_bNetPlay = true;
      s_recordingStartTime = Common::Timer::GetTimeSinceJan1970();
    }
    else if (Config::Get(Config::MAIN_CUSTOM_RTC_ENABLE))
    {
      s_recordingStartTime = Config::Get(Config::MAIN_CUSTOM_RTC_VALUE);
    }
    else
    {
      s_recordingStartTime = Common::Timer::GetLocalTimeSinceJan1970();
    }

    s_rerecords = 0;

    for (int i = 0; i < SerialInterface::MAX_SI_CHANNELS; ++i)
    {
      const SerialInterface::SIDevices si_device = Config::Get(Config::GetInfoForSIDevice(i));
      if (si_device == SerialInterface::SIDEVICE_GC_TARUKONGA)
        s_bongos |= (1 << i);
    }

    if (Core::IsRunningAndStarted())
    {
      const std::string save_path = File::GetUserPath(D_STATESAVES_IDX) + "dtm.sav";
      if (File::Exists(save_path))
        File::Delete(save_path);

      State::SaveAs(save_path);
      s_bRecordingFromSaveState = true;

      std::thread md5thread(GetMD5);
      md5thread.detach();
      GetSettings();
    }

    // Wiimotes cause desync issues if they're not reset before launching the game
    if (!Core::IsRunningAndStarted())
    {
      // This will also reset the Wiimotes for GameCube games, but that shouldn't do anything
      Wiimote::ResetAllWiimotes();
    }

    s_playMode = PlayMode::Recording;
    s_author = Config::Get(Config::MAIN_MOVIE_MOVIE_AUTHOR);
    s_temp_input.clear();

    s_currentByte = 0;

    if (Core::IsRunning())
      Core::UpdateWantDeterminism();
  });

  Core::DisplayMessage("Starting movie recording", 2000);
  return true;
}

static std::string Analog2DToString(u32 x, u32 y, const std::string& prefix, u32 range = 255)
{
  const u32 center = range / 2 + 1;

  if ((x <= 1 || x == center || x >= range) && (y <= 1 || y == center || y >= range))
  {
    if (x != center || y != center)
    {
      if (x != center && y != center)
      {
        return fmt::format("{}:{},{}", prefix, x < center ? "LEFT" : "RIGHT",
                           y < center ? "DOWN" : "UP");
      }

      if (x != center)
      {
        return fmt::format("{}:{}", prefix, x < center ? "LEFT" : "RIGHT");
      }

      return fmt::format("{}:{}", prefix, y < center ? "DOWN" : "UP");
    }

    return "";
  }

  return fmt::format("{}:{},{}", prefix, x, y);
}

static std::string Analog1DToString(u32 v, const std::string& prefix, u32 range = 255)
{
  if (v == 0)
    return "";

  if (v == range)
    return prefix;

  return fmt::format("{}:{}", prefix, v);
}

// L + R + Y resets to the default savestate
static void handleCustomTrainingModeInput(ControllerState padState)
{
  if ((padState.TriggerL == 255 || padState.L) && (padState.TriggerR == 255 || padState.R) &&
      padState.Y)
  {
    StateAuxillary::loadStateFromTrainingBuffer();
  }
}

// NOTE: CPU Thread
static void SetInputDisplayString(ControllerState padState, int controllerID)
{
  std::string display_str = fmt::format("P{}:", controllerID + 1);

  if (padState.is_connected)
  {
    if (padState.A)
      display_str += " A";
    if (padState.B)
      display_str += " B";
    if (padState.X)
      display_str += " X";
    if (padState.Y)
      display_str += " Y";
    if (padState.Z)
      display_str += " Z";
    if (padState.Start)
      display_str += " START";

    if (padState.DPadUp)
      display_str += " UP";
    if (padState.DPadDown)
      display_str += " DOWN";
    if (padState.DPadLeft)
      display_str += " LEFT";
    if (padState.DPadRight)
      display_str += " RIGHT";
    if (padState.reset)
      display_str += " RESET";

    if (padState.TriggerL == 255 || padState.L)
      display_str += " L";
    else
      display_str += Analog1DToString(padState.TriggerL, " L");

    if (padState.TriggerR == 255 || padState.R)
      display_str += " R";
    else
      display_str += Analog1DToString(padState.TriggerR, " R");

    display_str += Analog2DToString(padState.AnalogStickX, padState.AnalogStickY, " ANA");
    display_str += Analog2DToString(padState.CStickX, padState.CStickY, " C");

    // hook into buttons for manually resetting training mode
    if (StateAuxillary::getCustomTrainingModeStart())
    {
      Movie::handleCustomTrainingModeInput(padState);
    }
  }
  else
  {
    display_str += " DISCONNECTED";
  }

  std::lock_guard guard(s_input_display_lock);
  s_InputDisplay[controllerID] = std::move(display_str);
}

// NOTE: CPU Thread
static void SetWiiInputDisplayString(int remoteID, const DataReportBuilder& rpt, int ext,
                                     const EncryptionKey& key)
{
  int controllerID = remoteID + 4;

  std::string display_str = fmt::format("R{}:", remoteID + 1);

  if (rpt.HasCore())
  {
    ButtonData buttons;
    rpt.GetCoreData(&buttons);

    if (buttons.left)
      display_str += " LEFT";
    if (buttons.right)
      display_str += " RIGHT";
    if (buttons.down)
      display_str += " DOWN";
    if (buttons.up)
      display_str += " UP";
    if (buttons.a)
      display_str += " A";
    if (buttons.b)
      display_str += " B";
    if (buttons.plus)
      display_str += " +";
    if (buttons.minus)
      display_str += " -";
    if (buttons.one)
      display_str += " 1";
    if (buttons.two)
      display_str += " 2";
    if (buttons.home)
      display_str += " HOME";
  }

  if (rpt.HasAccel())
  {
    AccelData accel_data;
    rpt.GetAccelData(&accel_data);

    // FYI: This will only print partial data for interleaved reports.

    display_str +=
        fmt::format(" ACC:{},{},{}", accel_data.value.x, accel_data.value.y, accel_data.value.z);
  }

  if (rpt.HasIR())
  {
    const u8* const ir_data = rpt.GetIRDataPtr();

    // TODO: This does not handle the different IR formats.

    const u16 x = ir_data[0] | ((ir_data[2] >> 4 & 0x3) << 8);
    const u16 y = ir_data[1] | ((ir_data[2] >> 6 & 0x3) << 8);
    display_str += fmt::format(" IR:{},{}", x, y);
  }

  // Nunchuk
  if (rpt.HasExt() && ext == ExtensionNumber::NUNCHUK)
  {
    const u8* const extData = rpt.GetExtDataPtr();

    Nunchuk::DataFormat nunchuk;
    memcpy(&nunchuk, extData, sizeof(nunchuk));
    key.Decrypt((u8*)&nunchuk, 0, sizeof(nunchuk));
    nunchuk.bt.hex = nunchuk.bt.hex ^ 0x3;

    const std::string accel = fmt::format(" N-ACC:{},{},{}", nunchuk.GetAccelX(),
                                          nunchuk.GetAccelY(), nunchuk.GetAccelZ());

    if (nunchuk.bt.c)
      display_str += " C";
    if (nunchuk.bt.z)
      display_str += " Z";
    display_str += accel;
    display_str += Analog2DToString(nunchuk.jx, nunchuk.jy, " ANA");
  }

  // Classic controller
  if (rpt.HasExt() && ext == ExtensionNumber::CLASSIC)
  {
    const u8* const extData = rpt.GetExtDataPtr();

    Classic::DataFormat cc;
    memcpy(&cc, extData, sizeof(cc));
    key.Decrypt((u8*)&cc, 0, sizeof(cc));
    cc.bt.hex = cc.bt.hex ^ 0xFFFF;

    if (cc.bt.dpad_left)
      display_str += " LEFT";
    if (cc.bt.dpad_right)
      display_str += " RIGHT";
    if (cc.bt.dpad_down)
      display_str += " DOWN";
    if (cc.bt.dpad_up)
      display_str += " UP";
    if (cc.bt.a)
      display_str += " A";
    if (cc.bt.b)
      display_str += " B";
    if (cc.bt.x)
      display_str += " X";
    if (cc.bt.y)
      display_str += " Y";
    if (cc.bt.zl)
      display_str += " ZL";
    if (cc.bt.zr)
      display_str += " ZR";
    if (cc.bt.plus)
      display_str += " +";
    if (cc.bt.minus)
      display_str += " -";
    if (cc.bt.home)
      display_str += " HOME";

    display_str += Analog1DToString(cc.GetLeftTrigger().value, " L", 31);
    display_str += Analog1DToString(cc.GetRightTrigger().value, " R", 31);

    const auto left_stick = cc.GetLeftStick().value;
    display_str += Analog2DToString(left_stick.x, left_stick.y, " ANA", 63);

    const auto right_stick = cc.GetRightStick().value;
    display_str += Analog2DToString(right_stick.x, right_stick.y, " R-ANA", 31);
  }

  std::lock_guard guard(s_input_display_lock);
  s_InputDisplay[controllerID] = std::move(display_str);
}

// NOTE: CPU Thread
void CheckPadStatus(const GCPadStatus* PadStatus, int controllerID)
{
  s_padState.A = ((PadStatus->button & PAD_BUTTON_A) != 0);
  s_padState.B = ((PadStatus->button & PAD_BUTTON_B) != 0);
  s_padState.X = ((PadStatus->button & PAD_BUTTON_X) != 0);
  s_padState.Y = ((PadStatus->button & PAD_BUTTON_Y) != 0);
  s_padState.Z = ((PadStatus->button & PAD_TRIGGER_Z) != 0);
  s_padState.Start = ((PadStatus->button & PAD_BUTTON_START) != 0);

  s_padState.DPadUp = ((PadStatus->button & PAD_BUTTON_UP) != 0);
  s_padState.DPadDown = ((PadStatus->button & PAD_BUTTON_DOWN) != 0);
  s_padState.DPadLeft = ((PadStatus->button & PAD_BUTTON_LEFT) != 0);
  s_padState.DPadRight = ((PadStatus->button & PAD_BUTTON_RIGHT) != 0);

  s_padState.L = ((PadStatus->button & PAD_TRIGGER_L) != 0);
  s_padState.R = ((PadStatus->button & PAD_TRIGGER_R) != 0);
  s_padState.TriggerL = PadStatus->triggerLeft;
  s_padState.TriggerR = PadStatus->triggerRight;

  s_padState.AnalogStickX = PadStatus->stickX;
  s_padState.AnalogStickY = PadStatus->stickY;

  s_padState.CStickX = PadStatus->substickX;
  s_padState.CStickY = PadStatus->substickY;

  s_padState.is_connected = PadStatus->isConnected;

  s_padState.get_origin = (PadStatus->button & PAD_GET_ORIGIN) != 0;

  s_padState.disc = s_bDiscChange;
  s_bDiscChange = false;
  s_padState.reset = s_bReset;
  s_bReset = false;

  SetInputDisplayString(s_padState, controllerID);
}

// NOTE: CPU Thread
void RecordInput(const GCPadStatus* PadStatus, int controllerID)
{
  if (!IsRecordingInput() || !IsUsingPad(controllerID))
    return;

  CheckPadStatus(PadStatus, controllerID);

  // Cache the pad status for CITF capture
  s_last_pad_status[controllerID] = *PadStatus;

  s_temp_input.resize(s_currentByte + sizeof(ControllerState));
  memcpy(&s_temp_input[s_currentByte], &s_padState, sizeof(ControllerState));
  s_currentByte += sizeof(ControllerState);
}

// NOTE: CPU Thread
void CheckWiimoteStatus(int wiimote, const DataReportBuilder& rpt, int ext,
                        const EncryptionKey& key)
{
  SetWiiInputDisplayString(wiimote, rpt, ext, key);

  if (IsRecordingInput())
    RecordWiimote(wiimote, rpt.GetDataPtr(), rpt.GetDataSize());
}

void RecordWiimote(int wiimote, const u8* data, u8 size)
{
  if (!IsRecordingInput() || !IsUsingWiimote(wiimote))
    return;

  InputUpdate();
  s_temp_input.resize(s_currentByte + size + 1);
  s_temp_input[s_currentByte++] = size;
  memcpy(&s_temp_input[s_currentByte], data, size);
  s_currentByte += size;
}

// NOTE: EmuThread / Host Thread
void ReadHeader()
{
  for (int i = 0; i < 4; ++i)
  {
    if (tmpHeader.GBAControllers & (1 << i))
      s_controllers[i] = ControllerType::GBA;
    else if (tmpHeader.controllers & (1 << i))
      s_controllers[i] = ControllerType::GC;
    else
      s_controllers[i] = ControllerType::None;
    s_wiimotes[i] = (tmpHeader.controllers & (1 << (i + 4))) != 0;
  }
  s_recordingStartTime = tmpHeader.recordingStartTime;
  if (s_rerecords < tmpHeader.numRerecords)
    s_rerecords = tmpHeader.numRerecords;

  if (tmpHeader.bSaveConfig)
  {
    s_bSaveConfig = true;
    Config::AddLayer(ConfigLoaders::GenerateMovieConfigLoader(&tmpHeader));
    s_bClearSave = tmpHeader.bClearSave;
    s_memcards = tmpHeader.memcards;
    s_bongos = tmpHeader.bongos;
    s_bNetPlay = tmpHeader.bNetPlay;
    s_revision = tmpHeader.revision;
  }
  else
  {
    GetSettings();
  }

  s_discChange = {tmpHeader.discChange.begin(), tmpHeader.discChange.end()};
  s_author = {tmpHeader.author.begin(), tmpHeader.author.end()};
  s_MD5 = tmpHeader.md5;
  s_DSPiromHash = tmpHeader.DSPiromHash;
  s_DSPcoefHash = tmpHeader.DSPcoefHash;

  for (int i = 0; i < 4; i++)
  {
    Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(i)), SerialInterface::SIDevices(tmpHeader.reserved2[i+4]));
  }
  SConfig::GetInstance().SaveSettings();
  /*
  const SerialInterface::SIDevices gcnAdapter = SerialInterface::SIDEVICE_WIIU_ADAPTER;
  if (tmpHeader.bNetPlay)
  {
    // update controller settings via RESERVED field if we did netplay

    for (int i = 0; i < 4; i++)
    {
      if (tmpHeader.reserved[0] == i)
      {
        Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(i)), gcnAdapter);
      }
      else
      {
        Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(i)),
                                 SerialInterface::SIDEVICE_NONE);
      }
    }
  }
  else
  {
    // update controller settings via CONTROLLERS field
    // they might have plaeyd locally as p4, so we need only P4 to be on (example)
    // this works, but it's kinda up in the air as to why
    // also, if someone pauses mid game and has excess ports on they will get polling hell
    // encourage users to only have ports on that they need when playing locally
    for (int i = 0; i < 4; i++)
    {
      bool setControllerYet = false;
      if (tmpHeader.controllers & (1 << i))
      {
        if (!setControllerYet)
        {
          Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(i)), gcnAdapter);
          setControllerYet = true;
        }
        else
        {
          Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(i)),
                                   SerialInterface::SIDEVICE_NONE);
        }
      }
      else
      {
        Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(i)),
                                 SerialInterface::SIDEVICE_NONE);
      }
    }
  }
  SConfig::GetInstance().SaveSettings();
  */
}

#define dir_delimter '/'
#define MAX_FILENAME 512
#define READ_SIZE 8192

// Strip locale-formatted commas from numeric strings (e.g. "1,751,851,212" -> "1751851212")
static std::string StripCommas(const std::string& s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    if (c != ',')
      out += c;
  return out;
}

// Parse CIT output.json text into a CaptureMatchInfo struct
static CaptureMatchInfo ParseCITJson(const std::string& json_text)
{
  CaptureMatchInfo info;
  picojson::value v;
  std::string err = picojson::parse(v, json_text);
  if (!err.empty() || !v.is<picojson::object>())
  {
    WARN_LOG_FMT(CORE, "CIT JSON parse error: {}", err);
    return info;
  }
  const auto& obj = v.get<picojson::object>();

  auto getStr = [&](const std::string& key) -> std::string {
    auto it = obj.find(key);
    return (it != obj.end() && it->second.is<std::string>()) ? it->second.get<std::string>() : "";
  };

  // epoch ("1,751,851,212" -> u64)
  std::string epoch_str = StripCommas(getStr("Epoch"));
  if (!epoch_str.empty())
    info.epoch = std::stoull(epoch_str);

  // roomId ("68485da2" -> u32)
  std::string rid = getStr("Room ID");
  if (!rid.empty() && rid != "Empty")
    info.roomId = static_cast<u32>(std::stoul(rid, nullptr, 16));

  // gameCount ("1" -> u16)
  std::string gc_str = StripCommas(getStr("Game Count"));
  if (!gc_str.empty())
    info.gameCount = static_cast<u16>(std::stoul(gc_str));

  // citrusGameId ("68485da201" -> u64)
  std::string cgid = getStr("Citrus Game Id");
  if (!cgid.empty())
    info.citrusGameId = std::stoull(cgid, nullptr, 16);

  // submittedByDiscordId ("372784675359424512" -> u64)
  std::string sb = getStr("submittedBy");
  if (!sb.empty() && sb != "Empty")
    info.submittedByDiscordId = std::stoull(sb);

  // md5 ("8788cfdf60258c975fcb8632eb295f58" -> u8[16])
  std::string hash = getStr("Game Hash");
  for (int i = 0; i < 16 && (i * 2 + 1) < static_cast<int>(hash.size()); i++)
    info.md5[i] = static_cast<u8>(std::stoul(hash.substr(i * 2, 2), nullptr, 16));

  // boolean flags
  info.isRanked           = (getStr("isRanked") == "1");
  info.isNetplay          = (getStr("Netplay Match") == "1");
  info.overtimeNotReached = (getStr("Overtime Not Reached") == "1");
  info.matchItems         = (getStr("Match Items") == "1");
  info.matchSuperStrikes  = (getStr("Match Super Strikes") == "1");
  info.matchBowserOrFTX   = (getStr("Match Bowser or FTX") == "1");

  // match settings
  std::string mta = StripCommas(getStr("Match Time Allotted"));
  if (!mta.empty())
    info.matchTimeAllotted = static_cast<u16>(std::stoul(mta));

  std::string diff = getStr("Match Difficulty");
  if (!diff.empty())
    info.matchDifficulty = static_cast<u8>(std::stoul(diff));

  std::string elapsed = getStr("Match Time Elapsed");
  if (!elapsed.empty())
    info.matchTimeElapsed = std::stof(elapsed);

  // Controller port team assignments from "Controller Port Info" object
  auto portIt = obj.find("Controller Port Info");
  if (portIt != obj.end() && portIt->second.is<picojson::object>())
  {
    const auto& ports = portIt->second.get<picojson::object>();
    for (int i = 0; i < 4; i++)
    {
      auto pit = ports.find("Controller Port " + std::to_string(i));
      if (pit != ports.end() && pit->second.is<double>())
      {
        int val = static_cast<int>(pit->second.get<double>());
        info.ports[i].team = (val == 0) ? 0 : (val == 1) ? 1 : 0xFF;
      }
      else
      {
        info.ports[i].team = 0xFF;
      }
    }
  }

  // Player info from "Left Team Player Info" and "Right Team Player Info" arrays.
  // Each entry is ["P1 - PoolBoi", "372784675359424512"].
  // The name encodes port: "P1 - ..." -> port 0, "P2 - ..." -> port 1, etc.
  auto parseTeamPlayers = [&](const std::string& key)
  {
    auto it = obj.find(key);
    if (it == obj.end() || !it->second.is<picojson::array>())
      return;
    for (const auto& entry : it->second.get<picojson::array>())
    {
      if (!entry.is<picojson::array>())
        continue;
      const auto& arr = entry.get<picojson::array>();
      if (arr.size() < 2)
        continue;
      if (!arr[0].is<std::string>() || !arr[1].is<std::string>())
        continue;

      std::string nameWithPort = arr[0].get<std::string>();  // "P1 - PoolBoi"
      std::string discordIdStr = arr[1].get<std::string>();  // "372784675359424512"

      // Extract port index from "P1 - " prefix
      int port = -1;
      if (nameWithPort.size() >= 3 && nameWithPort[0] == 'P' && nameWithPort[2] == ' ')
        port = nameWithPort[1] - '1';  // "P1" -> 0, "P2" -> 1, etc.
      if (port < 0 || port > 3)
        continue;

      // Strip "Px - " prefix (5 chars) to get raw display name
      std::string rawName = (nameWithPort.size() > 5) ? nameWithPort.substr(5) : nameWithPort;

      info.ports[port].displayName = rawName;
      if (!discordIdStr.empty() && discordIdStr != "Empty")
        info.ports[port].discordId = std::stoull(discordIdStr);
    }
  };
  parseTeamPlayers("Left Team Player Info");
  parseTeamPlayers("Right Team Player Info");

  return info;
}

// Load CITF file and extract controller inputs for playback.
// Supports both compressed (zstd) and uncompressed files — detected by first 4 bytes.
static bool LoadCITFInputs(const std::string& citf_path)
{
  // Read entire file into memory
  File::IOFile file(citf_path, "rb");
  if (!file)
  {
    WARN_LOG_FMT(CORE, "Failed to open CITF file: {}", citf_path);
    return false;
  }
  const u64 file_size = file.GetSize();
  std::vector<u8> file_buf(file_size);
  if (!file.ReadBytes(file_buf.data(), file_size))
  {
    ERROR_LOG_FMT(CORE, "Failed to read CITF file content");
    return false;
  }
  file.Close();

  // Detect and decompress zstd frames (magic bytes: 0xFD 0x2F 0xB5 0x28)
  std::vector<u8> decomp_buf;
  const u8* data = file_buf.data();
  size_t data_size = static_cast<size_t>(file_size);

  if (file_size >= 4 && data[0] == 0x28 && data[1] == 0xB5 &&
      data[2] == 0x2F && data[3] == 0xFD)
  {
    const unsigned long long content_size =
        ZSTD_getFrameContentSize(data, data_size);
    if (content_size == ZSTD_CONTENTSIZE_ERROR || content_size == ZSTD_CONTENTSIZE_UNKNOWN)
    {
      ERROR_LOG_FMT(CORE, "CITF: Could not determine decompressed size");
      return false;
    }
    decomp_buf.resize(static_cast<size_t>(content_size));
    const size_t result = ZSTD_decompress(decomp_buf.data(), content_size, data, data_size);
    if (ZSTD_isError(result))
    {
      ERROR_LOG_FMT(CORE, "CITF: zstd decompression failed: {}", ZSTD_getErrorName(result));
      return false;
    }
    INFO_LOG_FMT(CORE, "CITF: Decompressed {} -> {} bytes ({:.1f}x ratio)",
                 file_size, result, static_cast<float>(result) / file_size);
    data = decomp_buf.data();
    data_size = result;
  }

  // 48-byte base header (layout identical from v7 through v11+)
  struct CITFHeader
  {
    char magic[4];
    u32 version;
    u32 frameCount;
    u32 fixedFrameSize;
    u8 leftCaptainID;
    u8 rightCaptainID;
    u8 leftSidekickID;
    u8 rightSidekickID;
    u8 stadiumID;
    u8 headerPadding[3];
    float goalLineX;
    float sidelineY;
    float penaltyBoxX;
    float netHalfWidth;
    float netHeight;
    float netDepth;
  };

  if (data_size < sizeof(CITFHeader))
  {
    ERROR_LOG_FMT(CORE, "CITF: Buffer too small for header ({} bytes)", data_size);
    return false;
  }

  CITFHeader header;
  std::memcpy(&header, data, sizeof(header));

  if (std::memcmp(header.magic, "CITF", 4) != 0)
  {
    ERROR_LOG_FMT(CORE, "Invalid CITF magic: {}{}{}{}", header.magic[0], header.magic[1],
                  header.magic[2], header.magic[3]);
    return false;
  }

  INFO_LOG_FMT(CORE, "Loading CITF inputs: {} frames, fixedFrameSize={}, version={}",
               header.frameCount, header.fixedFrameSize, header.version);

  s_citf_inputs.resize(header.frameCount);

  // Frame structure offsets (computed from fixedFrameSize to support version changes):
  // itemCount is always the last 4 bytes of the fixed portion (u8 + 3 padding).
  // Working backwards from itemCount:
  //   v8+ FrameTeamStats block: leftStats(10) + rightStats(10) + statsPadding(4) = 24 bytes
  //   inventory: 4 slots x 8 bytes = 32 bytes
  //   controllers: 4 ports x 10 bytes = 40 bytes
  const size_t movie_frame_number_offset = 4;  // After gameTime (stable across versions)
  const size_t item_count_offset = header.fixedFrameSize - 4;
  const size_t team_stats_size = (header.version >= 8) ? 24 : 0;
  const size_t controller_offset = item_count_offset - team_stats_size - 32 - 40;

  // v11+ header is 273 bytes; v10 and earlier is 48 bytes
  size_t current_offset = (header.version >= 11) ? 273 : sizeof(CITFHeader);

  // FrameControllerInput layout (10 bytes, matches packed struct in GameStateFrame.h)
  struct CITFControllerInput
  {
    u16 buttons;
    u8 stickX;
    u8 stickY;
    u8 substickX;
    u8 substickY;
    u8 triggerLeft;
    u8 triggerRight;
    u8 isConnected;
    u8 padding;
  };

  for (u32 frame_idx = 0; frame_idx < header.frameCount; frame_idx++)
  {
    if (current_offset + header.fixedFrameSize > data_size)
    {
      ERROR_LOG_FMT(CORE, "CITF: Unexpected end of data at frame {}", frame_idx);
      return false;
    }

    // Read movieFrameNumber
    std::memcpy(&s_citf_inputs[frame_idx].movieFrameNumber,
                data + current_offset + movie_frame_number_offset, sizeof(u32));

    // Read all 4 controller inputs
    for (int port = 0; port < 4; port++)
    {
      CITFControllerInput citf_input;
      std::memcpy(&citf_input,
                  data + current_offset + controller_offset + port * sizeof(CITFControllerInput),
                  sizeof(CITFControllerInput));

      GCPadStatus& pad = s_citf_inputs[frame_idx].controllers[port];
      pad.button = citf_input.buttons;
      pad.stickX = citf_input.stickX;
      pad.stickY = citf_input.stickY;
      pad.substickX = citf_input.substickX;
      pad.substickY = citf_input.substickY;
      pad.triggerLeft = citf_input.triggerLeft;
      pad.triggerRight = citf_input.triggerRight;
      pad.isConnected = (citf_input.isConnected != 0);
      pad.analogA = (citf_input.buttons & 0x0100) ? 0xFF : 0;  // PAD_BUTTON_A
      pad.analogB = (citf_input.buttons & 0x0200) ? 0xFF : 0;  // PAD_BUTTON_B
    }

    // Advance to next frame (variable size due to items)
    const u8 item_count = data[current_offset + item_count_offset];
    current_offset += header.fixedFrameSize + item_count * 48;  // sizeof(FrameItem) = 48
  }

  INFO_LOG_FMT(CORE, "Successfully loaded {} frames of CITF inputs from {}", header.frameCount,
               citf_path);

  // Debug: verify movieFrameNumber progression on first 10 frames
  u32 prev_movie_frame = 0;
  bool is_sorted = true;
  for (u32 i = 0; i < std::min(10u, header.frameCount); i++)
  {
    const u32 curr = s_citf_inputs[i].movieFrameNumber;
    INFO_LOG_FMT(CORE, "  CITF frame {} -> movieFrameNumber={}", i, curr);
    if (i > 0 && curr <= prev_movie_frame)
    {
      ERROR_LOG_FMT(CORE, "  WARNING: movieFrameNumber not increasing! {} <= {}", curr,
                    prev_movie_frame);
      is_sorted = false;
    }
    prev_movie_frame = curr;
  }
  if (!is_sorted)
    ERROR_LOG_FMT(CORE, "CITF movieFrameNumbers are not sorted! Playback may be incorrect.");

  return true;
}

// NOTE: Host Thread
bool PlayInput(const std::string& movie_path, std::optional<std::string>* savestate_path)
{
  // we're going to set controllers on/off about 50 lines after this, so we need to
  // get the user's current ports so that we can set it back after the movie is closed out of

  const SerialInterface::SIDevices currentDevice0 =
      Config::Get(Config::GetInfoForSIDevice(static_cast<int>(0)));
  const SerialInterface::SIDevices currentDevice1 =
      Config::Get(Config::GetInfoForSIDevice(static_cast<int>(1)));
  const SerialInterface::SIDevices currentDevice2 =
      Config::Get(Config::GetInfoForSIDevice(static_cast<int>(2)));
  const SerialInterface::SIDevices currentDevice3 =
      Config::Get(Config::GetInfoForSIDevice(static_cast<int>(3)));
  StateAuxillary::setPrePort(currentDevice0, currentDevice1, currentDevice2, currentDevice3);

  // movie_path is a const and trying to change that breaks a lot of things
  std::string actual_movie_path = movie_path;

  if (s_playMode != PlayMode::None)
    return false;

  // check if it's a citrus playback file
  fs::path temp_movie_path = movie_path;
  if (temp_movie_path.extension() == ".cit")
  {
    s_cit_stem_name = temp_movie_path.stem().string();
    s_cit_dir_path  = temp_movie_path.parent_path().string();
    // unzip and store the cit file path to movie_path
    unzFile zipfile = unzOpen(movie_path.c_str());
    if (zipfile == NULL)
    {
      printf("not found");
    }

    bool boolFoundOutputSav = false;
    std::string json_file_path;  // path to extracted output.json, if found
    // Reset match info from any prior CIT before parsing this one
    GameStateCapture::SetMatchInfo(CaptureMatchInfo{});

    // Get info about the zip file
    unz_global_info global_info;
    if (unzGetGlobalInfo(zipfile, &global_info) != UNZ_OK)
    {
      printf("could not read file global info\n");
      unzClose(zipfile);
    }

    // Buffer to hold data read from the zip file.
    char read_buffer[READ_SIZE];

    // Loop to extract all files
    float i;
    for (i = 0; i < global_info.number_entry; ++i)
    {
      // Get info about current file.
      unz_file_info file_info;
      char filename[MAX_FILENAME];
      if (unzGetCurrentFileInfo(zipfile, &file_info, filename, MAX_FILENAME, NULL, 0, NULL, 0) !=
          UNZ_OK)
      {
        printf("could not read file info\n");
        unzClose(zipfile);
      }

      // Check if this entry is a directory or file.
      const size_t filename_length = strlen(filename);
      if (filename[filename_length - 1] == dir_delimter)
      {
        // Entry is a directory, so create it.
        printf("dir:%s\n", filename);
#ifdef _WIN32
        _mkdir(filename);
#else
        mkdir(filename, 0777);
#endif
      }
      else
      {
        // Entry is a file, so extract it.
        printf("file:%s\n", filename);

        // check if this is our movie (dtm file). if it is we need to make that our movie path
        fs::path foundDTMFile = filename;
        fs::path directory = fs::path(movie_path).parent_path();
        directory /= filename;
        std::string extractHere = directory.string();
        if (foundDTMFile.extension() == ".dtm")
        {
          // fs::path directory{movie_path};
          // std::string path_of_movie_path_string{path_of_movie_path.string()};

          actual_movie_path = directory.string();
        }
        if (foundDTMFile.extension() == ".sav")
        {
          INFO_LOG_FMT(CORE, "We found a savestate in the CIT");
          boolFoundOutputSav = true;
        }
        if (foundDTMFile.extension() == ".citframes")
        {
          INFO_LOG_FMT(CORE, "We found a CITF file in the CIT: {}", extractHere);
          s_citf_file_path = extractHere;
        }
        if (foundDTMFile.extension() == ".json")
        {
          INFO_LOG_FMT(CORE, "We found a JSON file in the CIT: {}", extractHere);
          json_file_path = extractHere;
        }
        if (unzOpenCurrentFile(zipfile) != UNZ_OK)
        {
          printf("could not open file\n");
          unzClose(zipfile);
        }

        // Open a file to write out the data.
        FILE* out = fopen(extractHere.c_str(), "wb");
        if (out == NULL)
        {
          printf("could not open destination file\n");
          unzCloseCurrentFile(zipfile);
          unzClose(zipfile);
        }

        int error = UNZ_OK;
        do
        {
          error = unzReadCurrentFile(zipfile, read_buffer, READ_SIZE);
          if (error < 0)
          {
            printf("error %d\n", error);
            unzCloseCurrentFile(zipfile);
            unzClose(zipfile);
          }

          // Write data to file.
          if (error > 0)
          {
            fwrite(read_buffer, error, 1, out);  // You should check return of fwrite...
          }
        } while (error > 0);

        fclose(out);
      }

      unzCloseCurrentFile(zipfile);

      // Go the the next entry listed in the zip file.
      if ((i + 1) < global_info.number_entry)
      {
        if (unzGoToNextFile(zipfile) != UNZ_OK)
        {
          printf("cound not read next file\n");
          unzClose(zipfile);
        }
      }
    }

    unzClose(zipfile);

    // Parse the CIT output.json (if found) to populate match metadata for the v11 CITF header
    if (!json_file_path.empty())
    {
      std::string json_text;
      if (File::ReadFileToString(json_file_path, json_text))
      {
        CaptureMatchInfo match_info = ParseCITJson(json_text);
        GameStateCapture::SetMatchInfo(match_info);
        INFO_LOG_FMT(CORE, "CIT JSON parsed: epoch={} roomId={:08X} gameCount={} "
                     "players: P0='{}' P1='{}' P2='{}' P3='{}'",
                     match_info.epoch, match_info.roomId, match_info.gameCount,
                     match_info.ports[0].displayName, match_info.ports[1].displayName,
                     match_info.ports[2].displayName, match_info.ports[3].displayName);
      }
      else
      {
        WARN_LOG_FMT(CORE, "Failed to read CIT JSON file: {}", json_file_path);
      }
    }

    // Check if a CITF file was found and load it for input playback
    if (!s_citf_file_path.empty())
    {
      INFO_LOG_FMT(CORE, "Attempting to load CITF inputs from: {}", s_citf_file_path);
      if (LoadCITFInputs(s_citf_file_path))
      {
        s_use_citf_inputs = true;
        INFO_LOG_FMT(CORE, "CITF playback mode enabled - using frame-level inputs via SI layer");
      }
      else
      {
        WARN_LOG_FMT(CORE, "Failed to load CITF inputs, falling back to DTM inputs");
        s_use_citf_inputs = false;
        s_citf_inputs.clear();
      }
    }

    // Load AI controller from INI config.  IPC port takes precedence over a
    // local ONNX model path, mirroring Movie::Init above.
    {
      int ai_ipc_port = Config::Get(Config::MAIN_MOVIE_AI_IPC_PORT);
      std::string ai_model_path = Config::Get(Config::MAIN_MOVIE_AI_MODEL_PATH);
      int ai_port = Config::Get(Config::MAIN_MOVIE_AI_CONTROLLED_PORT);
      bool ai_mirror_x = Config::Get(Config::MAIN_MOVIE_AI_MIRROR_X);
      if (ai_ipc_port > 0)
        InitAIControllerIpc(ai_ipc_port, ai_port, ai_mirror_x);
      else if (!ai_model_path.empty())
        InitAIController(ai_model_path, ai_port, ai_mirror_x);
    }

    if (!boolFoundOutputSav)
    {
      INFO_LOG_FMT(CORE, "We did not find a savestate in the CIT");
      // create output.dtm.sav from patch if we did not find it from unzipping

      fs::path incomingCITDirectory = fs::path(movie_path).parent_path();
      std::string baseSavPath = File::GetExeDirectory() + DIR_SEP + "base.sav";
      std::string diffFilePath = (incomingCITDirectory / "diffFile.patch").string();
      std::string outputSavPath = (incomingCITDirectory / "output.dtm.sav").string();

      if (File::Exists(diffFilePath) && File::Exists(baseSavPath))
      {
        // Read diff file
        std::ifstream diffStream(diffFilePath, std::ios::binary);
        std::vector<unsigned char> diffFileData((std::istreambuf_iterator<char>(diffStream)), {});
        diffStream.close();

        // New format: first 8 bytes are the new data size (u64 LE), rest is HDiffPatch diff.
        // Old format (hdiffz -c-lzma2): starts with 'H','D','I','F','F' ... (handled on Windows).
        const bool isNewFormat =
            diffFileData.size() >= 8 &&
            !(diffFileData.size() >= 5 && diffFileData[0] == 'H' && diffFileData[1] == 'D' &&
              diffFileData[2] == 'I' && diffFileData[3] == 'F' && diffFileData[4] == 'F');

        if (isNewFormat)
        {
          uint64_t newDataSize = 0;
          memcpy(&newDataSize, diffFileData.data(), 8);

          if (newDataSize > 0 && newDataSize < 256ULL * 1024 * 1024)
          {
            std::ifstream oldStream(baseSavPath, std::ios::binary);
            std::vector<unsigned char> oldData((std::istreambuf_iterator<char>(oldStream)), {});
            oldStream.close();

            std::vector<unsigned char> newData(static_cast<size_t>(newDataSize));
            const unsigned char* diffStart = diffFileData.data() + 8;
            const unsigned char* diffEnd = diffFileData.data() + diffFileData.size();

            if (patch(newData.data(), newData.data() + newDataSize, oldData.data(),
                      oldData.data() + oldData.size(), diffStart, diffEnd))
            {
              std::ofstream outStream(outputSavPath, std::ios::binary);
              outStream.write(reinterpret_cast<const char*>(newData.data()),
                              static_cast<std::streamsize>(newDataSize));
              outStream.close();
              INFO_LOG_FMT(CORE, "Applied diff patch via library, wrote {} bytes to {}",
                           newDataSize, outputSavPath);
            }
            else
            {
              WARN_LOG_FMT(CORE, "HDiffPatch patch() failed for {}", diffFilePath);
            }
          }
          else
          {
            WARN_LOG_FMT(CORE, "Diff file has invalid new data size: {}", newDataSize);
          }
        }
        else
        {
          // Old compressed diff format (created by hdiffz -c-lzma2).
          // Use HDiffPatch's LZMA2 decompressor plugin — works on all platforms.
          hpatch_compressedDiffInfo diffInfo;
          const unsigned char* diffBytes = diffFileData.data();
          const unsigned char* diffBytesEnd = diffBytes + diffFileData.size();
          if (getCompressedDiffInfo_mem(&diffInfo, diffBytes, diffBytesEnd) &&
              diffInfo.newDataSize > 0 && diffInfo.newDataSize < 256ULL * 1024 * 1024)
          {
            std::ifstream oldStream(baseSavPath, std::ios::binary);
            std::vector<unsigned char> oldData((std::istreambuf_iterator<char>(oldStream)), {});
            oldStream.close();
            std::vector<unsigned char> newData(static_cast<size_t>(diffInfo.newDataSize));
            if (patch_decompress_mem(newData.data(), newData.data() + diffInfo.newDataSize,
                                     oldData.data(), oldData.data() + oldData.size(),
                                     diffBytes, diffBytesEnd, &lzma2DecompressPlugin))
            {
              std::ofstream outStream(outputSavPath, std::ios::binary);
              outStream.write(reinterpret_cast<const char*>(newData.data()),
                              static_cast<std::streamsize>(diffInfo.newDataSize));
              outStream.close();
              INFO_LOG_FMT(CORE, "Applied old lzma2 diff via library: {} bytes output",
                           diffInfo.newDataSize);
            }
            else
            {
              WARN_LOG_FMT(CORE, "LZMA2 patch_decompress_mem failed for {}", diffFilePath);
            }
          }
          else
          {
            WARN_LOG_FMT(CORE, "Failed to read compressed diff info from {}", diffFilePath);
          }
        }
      }
      else
      {
        if (!File::Exists(baseSavPath))
          WARN_LOG_FMT(CORE, "Cannot apply diff: base.sav not found at {}", baseSavPath);
        else
          WARN_LOG_FMT(CORE, "Cannot apply diff: diffFile.patch not found at {}", diffFilePath);
      }
    }
  }

  File::IOFile recording_file(actual_movie_path, "rb");
  if (!recording_file.ReadArray(&tmpHeader, 1))
    return false;

  if (!IsMovieHeader(tmpHeader.filetype))
  {
    PanicAlertFmtT("Invalid recording file");
    return false;
  }

  ReadHeader();
  s_totalFrames = tmpHeader.frameCount;
  s_totalLagCount = tmpHeader.lagCount;
  s_totalInputCount = tmpHeader.inputCount;
  s_totalTickCount = tmpHeader.tickCount;
  s_currentFrame = 0;
  s_currentLagCount = 0;
  s_currentInputCount = 0;

  s_playMode = PlayMode::Playing;

  // Suppress panic alert dialogs during movie playback so batch conversion isn't interrupted
  Common::SetEnableAlert(false);

  // Wiimotes cause desync issues if they're not reset before launching the game
  Wiimote::ResetAllWiimotes();

  Core::UpdateWantDeterminism();

  s_temp_input.resize(recording_file.GetSize() - 256);
  recording_file.ReadBytes(s_temp_input.data(), s_temp_input.size());
  s_currentByte = 0;
  recording_file.Close();

  // Load savestate (and skip to frame data)
  if (tmpHeader.bFromSaveState && savestate_path)
  {
    const std::string savestate_path_temp = actual_movie_path + ".sav";
    if (File::Exists(savestate_path_temp))
      *savestate_path = savestate_path_temp;
    s_bRecordingFromSaveState = true;
    Movie::LoadInput(actual_movie_path);
  }

  return true;
}

void DoState(PointerWrap& p)
{
  // many of these could be useful to save even when no movie is active,
  // and the data is tiny, so let's just save it regardless of movie state.
  p.Do(s_currentFrame);
  p.Do(s_currentByte);
  p.Do(s_currentLagCount);
  p.Do(s_currentInputCount);
  p.Do(s_bPolled);
  p.Do(s_tickCountAtLastInput);
  // other variables (such as s_totalBytes and s_totalFrames) are set in LoadInput
}

// NOTE: Host Thread
void LoadInput(const std::string& movie_path)
{
  File::IOFile t_record;
  if (!t_record.Open(movie_path, "r+b"))
  {
    PanicAlertFmtT("Failed to read {0}", movie_path);
    EndPlayInput(false);
    return;
  }

  t_record.ReadArray(&tmpHeader, 1);

  if (!IsMovieHeader(tmpHeader.filetype))
  {
    PanicAlertFmtT("Savestate movie {0} is corrupted, movie recording stopping...", movie_path);
    EndPlayInput(false);
    return;
  }
  ReadHeader();
  if (!s_bReadOnly)
  {
    s_rerecords++;
    tmpHeader.numRerecords = s_rerecords;
    t_record.Seek(0, File::SeekOrigin::Begin);
    t_record.WriteArray(&tmpHeader, 1);
  }

  ChangePads();
  if (SConfig::GetInstance().bWii)
    ChangeWiiPads(true);

  u64 totalSavedBytes = t_record.GetSize() - 256;

  bool afterEnd = false;
  // This can only happen if the user manually deletes data from the dtm.
  if (s_currentByte > totalSavedBytes)
  {
    PanicAlertFmtT(
        "Warning: You loaded a save whose movie ends before the current frame in the save "
        "(byte {0} < {1}) (frame {2} < {3}). You should load another save before continuing.",
        totalSavedBytes + 256, s_currentByte + 256, tmpHeader.frameCount, s_currentFrame);
    afterEnd = true;
  }

  if (!s_bReadOnly || s_temp_input.empty())
  {
    s_totalFrames = tmpHeader.frameCount;
    s_totalLagCount = tmpHeader.lagCount;
    s_totalInputCount = tmpHeader.inputCount;
    s_totalTickCount = s_tickCountAtLastInput = tmpHeader.tickCount;

    s_temp_input.resize(static_cast<size_t>(totalSavedBytes));
    t_record.ReadBytes(s_temp_input.data(), s_temp_input.size());
  }
  else if (s_currentByte > 0)
  {
    if (s_currentByte > totalSavedBytes)
    {
    }
    else if (s_currentByte > s_temp_input.size())
    {
      afterEnd = true;
      PanicAlertFmtT(
          "Warning: You loaded a save that's after the end of the current movie. (byte {0} "
          "> {1}) (input {2} > {3}). You should load another save before continuing, or load "
          "this state with read-only mode off.",
          s_currentByte + 256, s_temp_input.size() + 256, s_currentInputCount, s_totalInputCount);
    }
    else if (s_currentByte > 0 && !s_temp_input.empty())
    {
      // verify identical from movie start to the save's current frame
      std::vector<u8> movInput(s_currentByte);
      t_record.ReadArray(movInput.data(), movInput.size());

      const auto result = std::mismatch(movInput.begin(), movInput.end(), s_temp_input.begin());

      if (result.first != movInput.end())
      {
        const ptrdiff_t mismatch_index = std::distance(movInput.begin(), result.first);

        // this is a "you did something wrong" alert for the user's benefit.
        // we'll try to say what's going on in excruciating detail, otherwise the user might not
        // believe us.
        if (IsUsingWiimote(0))
        {
          const size_t byte_offset = static_cast<size_t>(mismatch_index) + sizeof(DTMHeader);

          // TODO: more detail
          PanicAlertFmtT("Warning: You loaded a save whose movie mismatches on byte {0} ({1:#x}). "
                         "You should load another save before continuing, or load this state with "
                         "read-only mode off. Otherwise you'll probably get a desync.",
                         byte_offset, byte_offset);

          std::copy(movInput.begin(), movInput.end(), s_temp_input.begin());
        }
        else
        {
          const ptrdiff_t frame = mismatch_index / sizeof(ControllerState);
          ControllerState curPadState;
          memcpy(&curPadState, &s_temp_input[frame * sizeof(ControllerState)],
                 sizeof(ControllerState));
          ControllerState movPadState;
          memcpy(&movPadState, &movInput[frame * sizeof(ControllerState)], sizeof(ControllerState));
          PanicAlertFmtT(
              "Warning: You loaded a save whose movie mismatches on frame {0}. You should load "
              "another save before continuing, or load this state with read-only mode off. "
              "Otherwise you'll probably get a desync.\n\n"
              "More information: The current movie is {1} frames long and the savestate's movie "
              "is {2} frames long.\n\n"
              "On frame {3}, the current movie presses:\n"
              "Start={4}, A={5}, B={6}, X={7}, Y={8}, Z={9}, DUp={10}, DDown={11}, DLeft={12}, "
              "DRight={13}, L={14}, R={15}, LT={16}, RT={17}, AnalogX={18}, AnalogY={19}, CX={20}, "
              "CY={21}, Connected={22}"
              "\n\n"
              "On frame {23}, the savestate's movie presses:\n"
              "Start={24}, A={25}, B={26}, X={27}, Y={28}, Z={29}, DUp={30}, DDown={31}, "
              "DLeft={32}, DRight={33}, L={34}, R={35}, LT={36}, RT={37}, AnalogX={38}, "
              "AnalogY={39}, CX={40}, CY={41}, Connected={42}",
              frame, s_totalFrames, tmpHeader.frameCount, frame, curPadState.Start, curPadState.A,
              curPadState.B, curPadState.X, curPadState.Y, curPadState.Z, curPadState.DPadUp,
              curPadState.DPadDown, curPadState.DPadLeft, curPadState.DPadRight, curPadState.L,
              curPadState.R, curPadState.TriggerL, curPadState.TriggerR, curPadState.AnalogStickX,
              curPadState.AnalogStickY, curPadState.CStickX, curPadState.CStickY,
              curPadState.is_connected, frame, movPadState.Start, movPadState.A, movPadState.B,
              movPadState.X, movPadState.Y, movPadState.Z, movPadState.DPadUp, movPadState.DPadDown,
              movPadState.DPadLeft, movPadState.DPadRight, movPadState.L, movPadState.R,
              movPadState.TriggerL, movPadState.TriggerR, movPadState.AnalogStickX,
              movPadState.AnalogStickY, movPadState.CStickX, movPadState.CStickY,
              curPadState.is_connected);
        }
      }
    }
  }
  t_record.Close();

  s_bSaveConfig = tmpHeader.bSaveConfig;

  if (!afterEnd)
  {
    if (s_bReadOnly)
    {
      if (s_playMode != PlayMode::Playing)
      {
        s_playMode = PlayMode::Playing;
        Core::UpdateWantDeterminism();
        Core::DisplayMessage("Switched to playback", 2000);
      }
    }
    else
    {
      if (s_playMode != PlayMode::Recording)
      {
        s_playMode = PlayMode::Recording;
        Core::UpdateWantDeterminism();
        Core::DisplayMessage("Switched to recording", 2000);
      }
    }
  }
  else
  {
    EndPlayInput(false);
  }
}

// NOTE: CPU Thread
static void CheckInputEnd()
{
  if (s_currentByte >= s_temp_input.size() ||
      (CoreTiming::GetTicks() > s_totalTickCount && !IsRecordingInputFromSaveState()))
  {
    EndPlayInput(!s_bReadOnly);
    // delete citrus residue if any
    std::thread t1(&StateAuxillary::endPlayback);
    t1.detach();
    StateAuxillary::setPostPort();
  }
}

// NOTE: CPU Thread
void PlayController(GCPadStatus* PadStatus, int controllerID)
{
  // AI Controller Mode: override inputs only during active match play (gamePhase 4/5).
  // OnFrameEnd() sets IsMatchActive() true only when inference ran successfully.
  // During menus / goal celebrations this is false and we fall through to normal input.
  static int s_ai_log_counter = 0;
  if (s_use_ai_inputs && s_ai_controller && !IsPlayingInput())
  {
    bool loaded   = s_ai_controller->IsLoaded();
    bool active   = s_ai_controller->IsMatchActive();
    bool port_ok  = (controllerID == s_ai_controlled_port);
    if (++s_ai_log_counter % 120 == 1)
    {
      INFO_LOG_FMT(CORE, "PlayController AI check: port={} loaded={} active={} port_ok={} phase={}",
                   controllerID, loaded, active, port_ok, AIController::GetGamePhase());
    }

    if (loaded && active && port_ok)
    {
      GCPadStatus out = s_ai_controller->GetLastOutput();
      if (s_ai_log_counter % 120 == 1)
      {
        INFO_LOG_FMT(CORE, "PlayController AI output: btn=0x{:04X} stickX={} stickY={}",
                     out.button, out.stickX, out.stickY);
      }
      *PadStatus = out;
      s_last_pad_status[controllerID] = *PadStatus;
      return;
    }
  }

  // Correct playback is entirely dependent on the emulator polling the controllers
  // in the same order done during recording
  if (!IsPlayingInput() || !IsUsingPad(controllerID))
    return;

  // CITF Playback Mode: Use frame-level inputs from CITF file instead of DTM
  // Feeds through SI layer (same as DTM) — no game memory injection needed
  if (s_use_citf_inputs && !s_citf_inputs.empty())
  {
    // Find the CITF frame with the largest movieFrameNumber <= s_currentInputCount
    // This handles the 2:1 SI poll mapping (two SI polls per game frame):
    //   CITF movieFrameNumbers: 2, 4, 6, ...
    //   SI inputCounts:         2, 3, 4, 5, 6, 7, ...
    //   inputCount=2 -> CITF frame 2, inputCount=3 -> CITF frame 2 (reused)
    auto it = std::upper_bound(
        s_citf_inputs.begin(), s_citf_inputs.end(), s_currentInputCount,
        [](u64 target, const CITFFrameInputs& frame) {
          return target < frame.movieFrameNumber;
        });

    // upper_bound gives first element > target; we want the one before it
    const CITFFrameInputs* matching_frame =
        (it != s_citf_inputs.begin()) ? &*(it - 1) : nullptr;

    if (matching_frame == nullptr)
    {
      // No CITF frame available yet for this input count
      // This can happen during the initial frames before CITF capture started
      static int no_frame_warning_count = 0;
      if (no_frame_warning_count < 10)
      {
        WARN_LOG_FMT(CORE, "No CITF frame available for inputCount={} frame={} (first CITF movieFrameNumber={}), using default input",
                     s_currentInputCount, s_currentFrame,
                     s_citf_inputs.empty() ? 0 : s_citf_inputs[0].movieFrameNumber);
        no_frame_warning_count++;
      }

      // Use a neutral input (centered stick, no buttons)
      std::memset(PadStatus, 0, sizeof(GCPadStatus));
      PadStatus->stickX = 128;
      PadStatus->stickY = 128;
      PadStatus->substickX = 128;
      PadStatus->substickY = 128;

      s_last_pad_status[controllerID] = *PadStatus;
      s_currentByte += sizeof(ControllerState);
      return;
    }

    // Use the matching frame's input — feed directly through SI layer
    // (no game memory injection needed; SI register emulation delivers this to the game)
    const GCPadStatus& citf_pad = matching_frame->controllers[controllerID];
    *PadStatus = citf_pad;

    // Cache for GetLastPadStatus()
    s_last_pad_status[controllerID] = *PadStatus;

    // Increment byte counter to keep DTM system happy
    s_currentByte += sizeof(ControllerState);

    // Debug logging
    static int citf_debug_count = 0;
    if (citf_debug_count < 20 && controllerID == 0)
    {
      INFO_LOG_FMT(CORE,
                   "[CITF PLAY] inputCount={} frame={} citfMovieFrame={} btn=0x{:04X} stick=({},{}) cstick=({},{}) L={} R={}",
                   s_currentInputCount, s_currentFrame, matching_frame->movieFrameNumber,
                   PadStatus->button, PadStatus->stickX, PadStatus->stickY,
                   PadStatus->substickX, PadStatus->substickY,
                   PadStatus->triggerLeft, PadStatus->triggerRight);
      citf_debug_count++;
    }

    return;
  }

  // Standard DTM Playback Mode: Use SI poll-level inputs from DTM file
  if (s_temp_input.empty())
    return;

  if (s_currentByte + sizeof(ControllerState) > s_temp_input.size())
  {
    PanicAlertFmtT("Premature movie end in PlayController. {0} + {1} > {2}", s_currentByte,
                   sizeof(ControllerState), s_temp_input.size());
    EndPlayInput(!s_bReadOnly);
    return;
  }

  memcpy(&s_padState, &s_temp_input[s_currentByte], sizeof(ControllerState));
  s_currentByte += sizeof(ControllerState);

  PadStatus->isConnected = s_padState.is_connected;

  PadStatus->triggerLeft = s_padState.TriggerL;
  PadStatus->triggerRight = s_padState.TriggerR;

  PadStatus->stickX = s_padState.AnalogStickX;
  PadStatus->stickY = s_padState.AnalogStickY;

  PadStatus->substickX = s_padState.CStickX;
  PadStatus->substickY = s_padState.CStickY;

  PadStatus->button = 0;
  PadStatus->button |= PAD_USE_ORIGIN;

  if (s_padState.A)
  {
    PadStatus->button |= PAD_BUTTON_A;
    PadStatus->analogA = 0xFF;
  }
  if (s_padState.B)
  {
    PadStatus->button |= PAD_BUTTON_B;
    PadStatus->analogB = 0xFF;
  }
  if (s_padState.X)
    PadStatus->button |= PAD_BUTTON_X;
  if (s_padState.Y)
    PadStatus->button |= PAD_BUTTON_Y;
  if (s_padState.Z)
    PadStatus->button |= PAD_TRIGGER_Z;
  if (s_padState.Start)
    PadStatus->button |= PAD_BUTTON_START;

  if (s_padState.DPadUp)
    PadStatus->button |= PAD_BUTTON_UP;
  if (s_padState.DPadDown)
    PadStatus->button |= PAD_BUTTON_DOWN;
  if (s_padState.DPadLeft)
    PadStatus->button |= PAD_BUTTON_LEFT;
  if (s_padState.DPadRight)
    PadStatus->button |= PAD_BUTTON_RIGHT;

  if (s_padState.L)
    PadStatus->button |= PAD_TRIGGER_L;
  if (s_padState.R)
    PadStatus->button |= PAD_TRIGGER_R;

  if (s_padState.get_origin)
    PadStatus->button |= PAD_GET_ORIGIN;

  if (s_padState.disc)
  {
    Core::RunAsCPUThread([] {
      if (!DVDInterface::AutoChangeDisc())
      {
        CPU::Break();
        PanicAlertFmtT("Change the disc to {0}", s_discChange);
      }
    });
  }

  if (s_padState.reset)
    ProcessorInterface::ResetButton_Tap();

  // Debug logging for DTM playback tracing
  static int play_debug = 0;
  if (controllerID == 0 && play_debug < 100)
  {
    INFO_LOG_FMT(CORE, "PlayController [DTM READ {}] inputCount={} Controller {}: button=0x{:04X} stick=({},{}) cstick=({},{}) L={} R={}",
                 play_debug, s_currentInputCount, controllerID, PadStatus->button,
                 PadStatus->stickX, PadStatus->stickY,
                 PadStatus->substickX, PadStatus->substickY,
                 PadStatus->triggerLeft, PadStatus->triggerRight);
    play_debug++;
  }

  // Cache the pad status for CITF capture
  s_last_pad_status[controllerID] = *PadStatus;

  SetInputDisplayString(s_padState, controllerID);
  CheckInputEnd();
}

// NOTE: CPU Thread
bool PlayWiimote(int wiimote, WiimoteCommon::DataReportBuilder& rpt, int ext,
                 const EncryptionKey& key)
{
  if (!IsPlayingInput() || !IsUsingWiimote(wiimote) || s_temp_input.empty())
    return false;

  if (s_currentByte > s_temp_input.size())
  {
    PanicAlertFmtT("Premature movie end in PlayWiimote. {0} > {1}", s_currentByte,
                   s_temp_input.size());
    EndPlayInput(!s_bReadOnly);
    return false;
  }

  const u8 size = rpt.GetDataSize();
  const u8 sizeInMovie = s_temp_input[s_currentByte];

  if (size != sizeInMovie)
  {
    PanicAlertFmtT(
        "Fatal desync. Aborting playback. (Error in PlayWiimote: {0} != {1}, byte {2}.){3}",
        sizeInMovie, size, s_currentByte,
        (s_controllers == ControllerTypeArray{}) ?
            " Try re-creating the recording with all GameCube controllers "
            "disabled (in Configure > GameCube > Device Settings)." :
            "");
    EndPlayInput(!s_bReadOnly);
    return false;
  }

  s_currentByte++;

  if (s_currentByte + size > s_temp_input.size())
  {
    PanicAlertFmtT("Premature movie end in PlayWiimote. {0} + {1} > {2}", s_currentByte, size,
                   s_temp_input.size());
    EndPlayInput(!s_bReadOnly);
    return false;
  }

  memcpy(rpt.GetDataPtr(), &s_temp_input[s_currentByte], size);
  s_currentByte += size;

  s_currentInputCount++;

  CheckInputEnd();
  return true;
}

// NOTE: Host / EmuThread / CPU Thread
void EndPlayInput(bool cont)
{
  if (cont)
  {
    // If !IsMovieActive(), changing s_playMode requires calling UpdateWantDeterminism
    ASSERT(IsMovieActive());

    s_playMode = PlayMode::Recording;
    // Keep alerts suppressed if an AI controller is still active (it manages its own restore).
    if (!s_use_ai_inputs)
      Common::SetEnableAlert(Config::Get(Config::MAIN_USE_PANIC_HANDLERS));
    Core::DisplayMessage("Reached movie end. Resuming recording.", 2000);
  }
  else if (s_playMode != PlayMode::None)
  {
    // We can be called by EmuThread during boot (CPU::State::PowerDown)
    bool was_running = Core::IsRunningAndStarted() && !CPU::IsStepping();
    if (was_running && Config::Get(Config::MAIN_MOVIE_PAUSE_MOVIE))
      CPU::Break();
    s_rerecords = 0;
    s_currentByte = 0;

    // Clean up CITF playback state
    if (s_use_citf_inputs)
    {
      INFO_LOG_FMT(CORE, "CITF playback ended, cleaning up");
      s_use_citf_inputs = false;
      s_citf_inputs.clear();
      s_citf_file_path.clear();
    }

    // Note: AI controller is intentionally NOT shut down here — it persists across
    // matches so back-to-back games don't reload the model. ShutdownAIController()
    // is called explicitly from Shutdown() or when the user disables it.

    if (s_playMode == PlayMode::Playing)
    {
      StateAuxillary::setPostPort();
    }
    s_playMode = PlayMode::None;
    // Keep alerts suppressed if an AI controller is still active (it manages its own restore).
    if (!s_use_ai_inputs)
      Common::SetEnableAlert(Config::Get(Config::MAIN_USE_PANIC_HANDLERS));
    Core::DisplayMessage("Movie End.", 2000);
    s_bRecordingFromSaveState = false;
    // we don't clear these things because otherwise we can't resume playback if we load a movie
    // state later
    // s_totalFrames = s_totalBytes = 0;
    // delete tmpInput;
    // tmpInput = nullptr;

    Core::QueueHostJob([=] { Core::UpdateWantDeterminism(); });
  }
}



// NOTE: Save State + Host Thread
void SaveRecording(const std::string& filename)
{
  File::IOFile save_record(filename, "wb");
  // Create the real header now and write it
  DTMHeader header;
  memset(&header, 0, sizeof(DTMHeader));

  header.filetype[0] = 'D';
  header.filetype[1] = 'T';
  header.filetype[2] = 'M';
  header.filetype[3] = 0x1A;
  strncpy(header.gameID.data(), SConfig::GetInstance().GetGameID().c_str(), 6);
  header.bWii = SConfig::GetInstance().bWii;
  header.controllers = 0;
  header.GBAControllers = 0;
  for (int i = 0; i < 4; ++i)
  {
    if (IsUsingGBA(i))
      header.GBAControllers |= 1 << i;
    if (IsUsingPad(i))
      header.controllers |= 1 << i;
    if (IsUsingWiimote(i) && SConfig::GetInstance().bWii)
      header.controllers |= 1 << (i + 4);
  }

  header.bFromSaveState = s_bRecordingFromSaveState;
  header.frameCount = s_totalFrames;
  header.lagCount = s_totalLagCount;
  header.inputCount = s_totalInputCount;
  header.numRerecords = s_rerecords;
  header.recordingStartTime = s_recordingStartTime;

  header.bSaveConfig = true;
  ConfigLoaders::SaveToDTM(&header);
  header.memcards = s_memcards;
  header.bClearSave = s_bClearSave;
  header.bNetPlay = s_bNetPlay;
  strncpy(header.discChange.data(), s_discChange.c_str(), header.discChange.size());
  strncpy(header.author.data(), s_author.c_str(), header.author.size());
  header.md5 = s_MD5;
  header.bongos = s_bongos;
  header.revision = s_revision;
  header.DSPiromHash = s_DSPiromHash;
  header.DSPcoefHash = s_DSPcoefHash;
  header.tickCount = s_totalTickCount;
  std::array<u8, 11> s_ourPortInfo;
  s_ourPortInfo.fill(0);
  int portCounter = 0;
  std::array<u8, 8> s_citrusGameId;
  s_citrusGameId.fill(0);
  if (NetPlay::IsNetPlayRunning())
  {
    std::vector<int> ourNetPlayPorts = StateAuxillary::getOurNetPlayPorts();
    for (int i = 0; i < 4; i++)
    {
      int portValue = ourNetPlayPorts.at(i);
      s_ourPortInfo[i] = portValue;
      if (portValue)
      {
        // we need to know what kind of controller they we were using for playback purposes
        // netplay uses your si device from port 1 (index 0) so we need to use a counter that tracks
        // how many ports we've used

        const SerialInterface::SIDevices currentDevice =
            Config::Get(Config::GetInfoForSIDevice(static_cast<int>(portCounter)));
        s_ourPortInfo[i + 4] = currentDevice;
        portCounter++;
      }
      else
      {
        s_ourPortInfo[i + 4] = 0;
      }
    }
    header.reserved2 = s_ourPortInfo;
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      if (IsUsingPad(i))
      {
        s_ourPortInfo[i] = 1;
        const SerialInterface::SIDevices currentDevice =
            Config::Get(Config::GetInfoForSIDevice(static_cast<int>(i)));
        s_ourPortInfo[i+4] = currentDevice;
      }
      else
      {
        s_ourPortInfo[i] = 0;
        s_ourPortInfo[i+4] = 0;
      }
    }
    header.reserved2 = s_ourPortInfo;
  }

  // At 0x25 and has a max length of 8
  // Citrus Game Id is either the Netplay Room ID with the game count appended or "FFFFFFFFFF"
  std::string citrusGameId = Metadata::getCitrusGameId();
  int indexCounter = 0;
  for (std::string::size_type i = 0; i < citrusGameId.size(); i += 2)
  {
    char first = citrusGameId[i];
    char second = citrusGameId[i + 1];
    std::string zeroX = "0x";
    std::string combinedString = zeroX + first + second;
    auto intRepresentation = std::stol(combinedString, nullptr, 0);
    s_citrusGameId[indexCounter] = intRepresentation;
    indexCounter++;
  }
  header.uniqueID = s_citrusGameId;
  // TODO
  // header.audioEmulator;

  save_record.WriteArray(&header, 1);
  bool success = save_record.WriteBytes(s_temp_input.data(), s_temp_input.size());
  if (success)
  {
    Core::DisplayMessage("DTM saved", 2000);
  }
  else
  {
    Core::DisplayMessage("Failed to save DTM to: " + filename, 2000);
  }

  if (success && s_bRecordingFromSaveState)
  {
    std::string stateFilename = filename + ".sav";
    success = File::Copy(File::GetUserPath(D_STATESAVES_IDX) + "dtm.sav", stateFilename);
  }

  if (success)
  {
    Core::DisplayMessage("Save State saved", 2000);
  }
  else
  {
    Core::DisplayMessage("Failed to save Save State to " + filename + ".sav", 2000);
  }
}

void SetGCInputManip(GCManipFunction func)
{
  s_gc_manip_func = std::move(func);
}
void SetWiiInputManip(WiiManipFunction func)
{
  s_wii_manip_func = std::move(func);
}

// NOTE: CPU Thread
void CallGCInputManip(GCPadStatus* PadStatus, int controllerID)
{
  if (s_gc_manip_func)
    s_gc_manip_func(PadStatus, controllerID);
}
// NOTE: CPU Thread
void CallWiiInputManip(DataReportBuilder& rpt, int controllerID, int ext, const EncryptionKey& key)
{
  if (s_wii_manip_func)
    s_wii_manip_func(rpt, controllerID, ext, key);
}

// NOTE: GPU Thread
void SetGraphicsConfig()
{
  g_Config.bEFBAccessEnable = tmpHeader.bEFBAccessEnable;
  g_Config.bSkipEFBCopyToRam = tmpHeader.bSkipEFBCopyToRam;
  g_Config.bEFBEmulateFormatChanges = tmpHeader.bEFBEmulateFormatChanges;
  g_Config.bImmediateXFB = tmpHeader.bImmediateXFB;
  g_Config.bSkipXFBCopyToRam = tmpHeader.bSkipXFBCopyToRam;
}

// NOTE: EmuThread / Host Thread
void GetSettings()
{
  using ExpansionInterface::EXIDeviceType;
  const EXIDeviceType slot_a_type = Config::Get(Config::MAIN_SLOT_A);
  const EXIDeviceType slot_b_type = Config::Get(Config::MAIN_SLOT_B);
  const bool slot_a_has_raw_memcard = slot_a_type == EXIDeviceType::MemoryCard;
  const bool slot_a_has_gci_folder = slot_a_type == EXIDeviceType::MemoryCardFolder;
  const bool slot_b_has_raw_memcard = slot_b_type == EXIDeviceType::MemoryCard;
  const bool slot_b_has_gci_folder = slot_b_type == EXIDeviceType::MemoryCardFolder;

  s_bSaveConfig = true;
  s_bNetPlay = NetPlay::IsNetPlayRunning();
  if (SConfig::GetInstance().bWii)
  {
    u64 title_id = SConfig::GetInstance().GetTitleID();
    s_bClearSave = !File::Exists(Common::GetTitleDataPath(title_id, Common::FROM_SESSION_ROOT) +
                                 "/banner.bin");
  }
  else
  {
    const auto gci_folder_has_saves = [](ExpansionInterface::Slot card_slot) {
      const auto [path, migrate] = ExpansionInterface::CEXIMemoryCard::GetGCIFolderPath(
          card_slot, ExpansionInterface::AllowMovieFolder::No);
      const u64 number_of_saves = File::ScanDirectoryTree(path, false).size;
      return number_of_saves > 0;
    };

    s_bClearSave =
        !(slot_a_has_raw_memcard && File::Exists(Config::Get(Config::MAIN_MEMCARD_A_PATH))) &&
        !(slot_b_has_raw_memcard && File::Exists(Config::Get(Config::MAIN_MEMCARD_B_PATH))) &&
        !(slot_a_has_gci_folder && gci_folder_has_saves(ExpansionInterface::Slot::A)) &&
        !(slot_b_has_gci_folder && gci_folder_has_saves(ExpansionInterface::Slot::B));
  }
  s_memcards |= (slot_a_has_raw_memcard || slot_a_has_gci_folder) << 0;
  s_memcards |= (slot_b_has_raw_memcard || slot_b_has_gci_folder) << 1;

  s_revision = ConvertGitRevisionToBytes(Common::GetScmRevGitStr());

  if (!Config::Get(Config::MAIN_DSP_HLE))
  {
    std::string irom_file = File::GetUserPath(D_GCUSER_IDX) + DSP_IROM;
    std::string coef_file = File::GetUserPath(D_GCUSER_IDX) + DSP_COEF;

    if (!File::Exists(irom_file))
      irom_file = File::GetSysDirectory() + GC_SYS_DIR DIR_SEP DSP_IROM;
    if (!File::Exists(coef_file))
      coef_file = File::GetSysDirectory() + GC_SYS_DIR DIR_SEP DSP_COEF;
    std::vector<u16> irom(DSP::DSP_IROM_SIZE);
    File::IOFile file_irom(irom_file, "rb");

    file_irom.ReadArray(irom.data(), irom.size());
    file_irom.Close();
    for (u16& entry : irom)
      entry = Common::swap16(entry);

    std::vector<u16> coef(DSP::DSP_COEF_SIZE);
    File::IOFile file_coef(coef_file, "rb");

    file_coef.ReadArray(coef.data(), coef.size());
    file_coef.Close();
    for (u16& entry : coef)
      entry = Common::swap16(entry);
    s_DSPiromHash =
        Common::HashAdler32(reinterpret_cast<u8*>(irom.data()), DSP::DSP_IROM_BYTE_SIZE);
    s_DSPcoefHash =
        Common::HashAdler32(reinterpret_cast<u8*>(coef.data()), DSP::DSP_COEF_BYTE_SIZE);
  }
  else
  {
    s_DSPiromHash = 0;
    s_DSPcoefHash = 0;
  }
}

static const mbedtls_md_info_t* s_md5_info = mbedtls_md_info_from_type(MBEDTLS_MD_MD5);

// NOTE: Entrypoint for own thread
static void CheckMD5()
{
  if (s_current_file_name.empty())
    return;

  for (int i = 0, n = 0; i < 16; ++i)
  {
    if (tmpHeader.md5[i] != 0)
      continue;
    n++;
    if (n == 16)
      return;
  }
  Core::DisplayMessage("Verifying checksum...", 2000);

  std::array<u8, 16> game_md5;
  mbedtls_md_file(s_md5_info, s_current_file_name.c_str(), game_md5.data());

  if (game_md5 == s_MD5)
    Core::DisplayMessage("Checksum of current game matches the recorded game.", 2000);
  else
    Core::DisplayMessage("Checksum of curs_playMode = PlayMode::Nonerent game does not match the recorded game!", 3000);
}

// NOTE: Entrypoint for own thread
static void GetMD5()
{
  if (s_current_file_name.empty())
    return;

  Core::DisplayMessage("Calculating checksum of game file...", 2000);
  mbedtls_md_file(s_md5_info, s_current_file_name.c_str(), s_MD5.data());
  Core::DisplayMessage("Finished calculating checksum.", 2000);
  Metadata::setMD5(s_MD5);
}

// NOTE: EmuThread
void Shutdown()
{
  s_currentInputCount = s_totalInputCount = s_totalFrames = s_tickCountAtLastInput = 0;
  s_temp_input.clear();

  // Clean up CITF playback state
  s_use_citf_inputs = false;
  s_citf_inputs.clear();
  s_citf_file_path.clear();

  // Clean up AI controller
  if (s_ai_controller)
  {
    s_ai_controller->Shutdown();
    s_ai_controller.reset();
  }
  s_use_ai_inputs = false;

  // shutdown is called any time the game (core) is closed
  // delete any residue from shutting down a playback early that wasn't handled from graceful movie end
  std::thread t1(&StateAuxillary::endPlayback);
  t1.detach();
}

void TickAIController()
{
  if (s_use_ai_inputs && s_ai_controller && s_ai_controller->IsLoaded())
    s_ai_controller->OnFrameEnd(s_ai_controlled_port, s_ai_mirror_x);
}

void InitAIController(const std::string& onnx_path, int controlled_port, bool mirror_x)
{
  if (onnx_path.empty())
    return;

  INFO_LOG_FMT(CORE, "AIController: initializing path='{}' port={} mirror={}", onnx_path,
               controlled_port, mirror_x);

  s_ai_controller = std::make_unique<AIController>();
  if (!s_ai_controller->Load(onnx_path))
  {
    ERROR_LOG_FMT(CORE, "AIController: failed to load model from {}", onnx_path);
    s_ai_controller.reset();
    s_use_ai_inputs = false;
    return;
  }

  s_ai_controlled_port = std::max(0, std::min(3, controlled_port));
  s_ai_mirror_x        = mirror_x;
  s_use_ai_inputs      = true;

  // Suppress panic alert dialogs during AI control. The model occasionally drives the game
  // into states that trip "Invalid read"/"Unable to resolve" panics, and the modal dialog
  // stalls the emulation thread until dismissed. DTM/CITF playback does the same in PlayInput().
  Common::SetEnableAlert(false);

  INFO_LOG_FMT(CORE, "AIController: active on port {} mirror_x={} model={}",
               s_ai_controlled_port, s_ai_mirror_x, onnx_path);
}

void InitAIControllerIpc(int ipc_port, int controlled_port, bool mirror_x)
{
  if (ipc_port <= 0 || ipc_port > 65535)
  {
    ERROR_LOG_FMT(CORE, "AIController: invalid IPC port {}", ipc_port);
    return;
  }

  INFO_LOG_FMT(CORE, "AIController: initializing IPC backend on port {} (gc_port={} mirror={})",
               ipc_port, controlled_port, mirror_x);

  s_ai_controller = std::make_unique<AIController>();

  // Reset callback: the IPC receiver thread will invoke this when the Python
  // trainer sends a RESET control message (typically after a match_end=1
  // STATE packet).  Maps savestate_id to a .sav next to the Dolphin binary
  // and loads it via Core::QueueHostJob — State::LoadAs is host-thread-only
  // and a direct call from the IPC receiver thread would deadlock (see
  // dolphin_pause_resume.md for the same gotcha on SetState).
  AIController::ResetCallback reset_cb = [](uint32_t savestate_id) {
    // Stadium-diversity savestates captured at first active-play frame after
    // kickoff: Daisy/Toad (P1, left) vs Peach/Toad (CPU, right), AI on port 0.
    static constexpr const char* kSavestateFiles[] = {
        "rl_palace.sav",
        "rl_underground.sav",
        "rl_battle_dome.sav",
    };
    constexpr uint32_t kNumSavestates =
        static_cast<uint32_t>(sizeof(kSavestateFiles) / sizeof(kSavestateFiles[0]));

    if (savestate_id >= kNumSavestates)
    {
      WARN_LOG_FMT(CORE, "AIController: RESET savestate_id={} out of range "
                         "[0,{}); ignoring",
                   savestate_id, kNumSavestates);
      return;
    }

    const std::string path = File::GetExeDirectory() + DIR_SEP +
                             kSavestateFiles[savestate_id];
    INFO_LOG_FMT(CORE, "AIController: RESET → loading savestate id={} path={}",
                 savestate_id, path);
    Core::QueueHostJob([path]() { State::LoadAs(path); });
  };

  if (!s_ai_controller->LoadIpc(ipc_port, std::move(reset_cb)))
  {
    ERROR_LOG_FMT(CORE, "AIController: failed to start IPC backend on port {}", ipc_port);
    s_ai_controller.reset();
    s_use_ai_inputs = false;
    return;
  }

  s_ai_controlled_port = std::max(0, std::min(3, controlled_port));
  s_ai_mirror_x        = mirror_x;
  s_use_ai_inputs      = true;

  // Suppress panic alert dialogs during AI/RL control (see InitAIController for rationale).
  // Especially important for headless RL runs where there is no Qt window to dismiss the modal.
  Common::SetEnableAlert(false);

  INFO_LOG_FMT(CORE, "AIController: IPC active on port {} (gc_port={} mirror={})",
               ipc_port, s_ai_controlled_port, s_ai_mirror_x);
}

void ShutdownAIController()
{
  if (s_ai_controller)
  {
    s_ai_controller->Shutdown();
    s_ai_controller.reset();
    // Restore panic alerts to the user's configured setting (we suppressed them on init).
    Common::SetEnableAlert(Config::Get(Config::MAIN_USE_PANIC_HANDLERS));
  }
  s_use_ai_inputs = false;
}

bool IsUsingAIInputs()
{
  return s_use_ai_inputs && s_ai_controller && s_ai_controller->IsLoaded();
}

}  // namespace Movie

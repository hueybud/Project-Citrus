#include "Core/StrikersPadHook.h"

#include <array>

#include "Common/Logging/Log.h"
#include "Core/HW/Memmap.h"
#include "Core/Movie.h"
#include "Core/PowerPC/PowerPC.h"
#include "InputCommon/GCPadStatus.h"

/*
This class is not being used right now
*/

namespace StrikersPadHook
{

// Static state
static std::array<PadData, 4> s_pad_data{};
static bool s_ai_mode = false;
static bool s_hook_active = false;

PadData GetPadData(int controller)
{
  if (controller < 0 || controller >= 4)
    return PadData{};
  return s_pad_data[controller];
}

void SetAIMode(bool enabled)
{
  s_ai_mode = enabled;
  if (enabled)
    INFO_LOG_FMT(CORE, "StrikersPadHook: AI mode enabled - will inject AI inputs");
  else
    INFO_LOG_FMT(CORE, "StrikersPadHook: AI mode disabled - capturing real inputs");
}

bool IsAIModeActive()
{
  return s_ai_mode;
}

// HLE hook function called at 0x801c3808
// This is called right before the game reads controller inputs
void HookFunction()
{
  // Read all 4 controllers from Dolphin's Movie/input system
  // This works for both DTM playback and live gameplay
  for (int i = 0; i < 4; i++)
  {
    GCPadStatus gc_pad = Movie::GetLastPadStatus(i);
    PadData& pad = s_pad_data[i];

    // GCPadStatus already uses u8 format (0-255, centered at 128)
    // Store directly - no conversion needed
    pad.buttons = gc_pad.button;
    pad.stickX = gc_pad.stickX;
    pad.stickY = gc_pad.stickY;
    pad.cStickX = gc_pad.substickX;
    pad.cStickY = gc_pad.substickY;
    pad.triggerL = gc_pad.triggerLeft;
    pad.triggerR = gc_pad.triggerRight;
    pad.analogA = gc_pad.analogA;
    pad.analogB = gc_pad.analogB;
    pad.error = gc_pad.isConnected ? 0 : 1;

    // Future: If AI mode is active, write AI-generated inputs to the PAD buffer
    if (s_ai_mode)
    {
      // TODO: Write AI inputs to game memory
      // u32 r13 = GPR(13);
      // u32 pad_buffer_ptr = Memory::Read_U32(r13 - 0x6050);
      // u32 pad_ptr = pad_buffer_ptr + (i * 0xC);
      // Memory::Write_U16(pad_ptr + 0, ai_buttons);
      // Memory::Write_U8(pad_ptr + 2, ai_stickX);
      // etc.
    }
  }

  // Debug logging for first few frames
  static int debug_count = 0;
  if (debug_count < 3)
  {
    INFO_LOG_FMT(CORE, "StrikersPadHook: Controller 0 - buttons=0x{:04X} stick=({},{}) cstick=({},{}) L={} R={} error={}",
                 s_pad_data[0].buttons,
                 s_pad_data[0].stickX, s_pad_data[0].stickY,
                 s_pad_data[0].cStickX, s_pad_data[0].cStickY,
                 s_pad_data[0].triggerL, s_pad_data[0].triggerR,
                 s_pad_data[0].error);
    debug_count++;
  }

  // DON'T skip the function - let it run normally
}

void Initialize()
{
  s_hook_active = false;
  s_ai_mode = false;
  s_pad_data = {};
  INFO_LOG_FMT(CORE, "StrikersPadHook: Initialized");
}

void Shutdown()
{
  s_hook_active = false;
  s_ai_mode = false;
  INFO_LOG_FMT(CORE, "StrikersPadHook: Shutdown");
}

}  // namespace StrikersPadHook

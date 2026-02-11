#include "GameStateFrame.h"

#include <cstddef>
#include <cstring>
#include <vector>

#include "Common/CommonPaths.h"
#include "Common/FileUtil.h"
#include "Common/IOFile.h"
#include "Common/Logging/Log.h"

#include "Core/HW/AddressSpace.h"
#include "Core/HW/GCPad.h"
#include "Core/HW/Memmap.h"
#include "Core/Metadata.h"
#include "Core/Movie.h"
#include "InputCommon/GCPadStatus.h"

// Static state
static std::vector<GameStateFrame> s_frame_buffer;
static bool s_capturing = false;

// Buffered controller inputs from last Movie::PlayController() call
struct BufferedInput
{
  GCPadStatus pad;
  u64 inputCount;
  bool valid;
};
static std::array<BufferedInput, 4> s_buffered_inputs{};

// Binary file format constants
static constexpr u32 CITF_VERSION = 4;

void GameStateCapture::BeginCapture()
{
  s_frame_buffer.clear();
  s_frame_buffer.reserve(20000);  // ~5 min at 60fps

  // Clear buffered inputs
  for (auto& input : s_buffered_inputs)
  {
    input.valid = false;
    input.inputCount = 0;
  }

  s_capturing = true;

  INFO_LOG_FMT(CORE, "GameStateCapture: Begin capture at Movie frame {}", Movie::GetCurrentFrame());
}

bool GameStateCapture::IsCapturing()
{
  return s_capturing;
}

void GameStateCapture::OnControllerInput(int port, const GCPadStatus& pad, u64 inputCount)
{
  if (port >= 0 && port < 4)
  {
    s_buffered_inputs[port].pad = pad;
    s_buffered_inputs[port].inputCount = inputCount;
    s_buffered_inputs[port].valid = true;
  }
}

void GameStateCapture::CaptureFrame()
{
  if (!s_capturing)
    return;

  GameStateFrame frame;
  std::memset(&frame, 0, sizeof(frame));

  const AddressSpace::Accessors* accessors =
      AddressSpace::GetAccessors(AddressSpace::Type::Effective);

  // Frame metadata (frameIndex is implicit from position in file)
  frame.gameTime = accessors->ReadF32(Metadata::addressTimeElapsed);
  // Use input count from SI callback (captured at PlayController time, always correct)
  frame.movieFrameNumber = s_buffered_inputs[0].valid
                               ? static_cast<u32>(s_buffered_inputs[0].inputCount)
                               : 0;

  // Score
  frame.leftScore = static_cast<u8>(Memory::Read_U16(Metadata::addressLeftSideScore));
  frame.rightScore = static_cast<u8>(Memory::Read_U16(Metadata::addressRightSideScore));
  frame.isPaused = Memory::Read_U8(Metadata::addressIsGamePaused);

  // Ball, characters, controllers, powerup inventory, items
  ReadBallState(frame);
  ReadCharacterState(frame);
  ReadControllerInputs(frame);
  ReadPowerupInventory(frame);
  ReadItems(frame);

  s_frame_buffer.push_back(frame);
}

void GameStateCapture::ReadBallState(GameStateFrame& frame)
{
  const AddressSpace::Accessors* accessors =
      AddressSpace::GetAccessors(AddressSpace::Type::Effective);

  // Ball position
  frame.ballPosX = accessors->ReadF32(Metadata::addressBallXPos);
  frame.ballPosY = accessors->ReadF32(Metadata::addressBallYPos);
  frame.ballPosZ = accessors->ReadF32(Metadata::addressBallZPos);

  // Ball velocity
  frame.ballVelX = accessors->ReadF32(Metadata::addressBallXVel);
  frame.ballVelY = accessors->ReadF32(Metadata::addressBallYVel);
  frame.ballVelZ = accessors->ReadF32(Metadata::addressBallZVel);

  // Ball ownership and perfect pass state
  u32 ball_ptr = Memory::Read_U32(Metadata::addressBallPointer);
  if (ball_ptr != 0)
  {
    frame.ballOwnerCharacterPointer = Memory::Read_U32(ball_ptr + 0x24);
    frame.isPerfectPass = Memory::Read_U8(ball_ptr + 0xa1);
  }
}

void GameStateCapture::ReadCharacterState(GameStateFrame& frame)
{
  const AddressSpace::Accessors* accessors =
      AddressSpace::GetAccessors(AddressSpace::Type::Effective);

  // Left strikers (indices 0-3)
  for (int i = 0; i < 4; i++)
  {
    u32 char_ptr = Memory::Read_U32(Metadata::addressCharacterPointersBase + (i * 4));
    if (char_ptr != 0)
    {
      frame.characters[i].posX = accessors->ReadF32(char_ptr + 0x520);
      frame.characters[i].posY = accessors->ReadF32(char_ptr + 0x524);
      frame.characters[i].actionState = Memory::Read_U32(char_ptr + 0x1d8);
      frame.characters[i].heading = Memory::Read_U16(char_ptr + 0x42);

      // Read effect type from char_ptr + 0x11C
      u32 effect_ptr = Memory::Read_U32(char_ptr + 0x11C);
      if (effect_ptr == Metadata::addressEffectFrozen)
        frame.characters[i].effectType = 1;
      else if (effect_ptr == Metadata::addressEffectOnFire)
        frame.characters[i].effectType = 2;
      else if (effect_ptr == Metadata::addressEffectStar)
        frame.characters[i].effectType = 3;
      else if (effect_ptr == Metadata::addressEffectElectrocuted)
        frame.characters[i].effectType = 4;
      else
        frame.characters[i].effectType = 0;  // No effect

      // Read speed item state from char_ptr + 0x370, 0x374, 0x36c
      s32 speed_item_type = static_cast<s32>(Memory::Read_U32(char_ptr + 0x370));
      frame.characters[i].speedItemType = (speed_item_type == -1) ? 0 : static_cast<u8>(speed_item_type);
      frame.characters[i].speedItemCount = static_cast<u8>(Memory::Read_U32(char_ptr + 0x374));
      frame.characters[i].speedItemTimer = accessors->ReadF32(char_ptr + 0x36c);
    }
  }

  // Left goalie (index 4)
  {
    u32 char_ptr = Memory::Read_U32(Metadata::addressLeftGoaliePointer);
    if (char_ptr != 0)
    {
      frame.characters[4].posX = accessors->ReadF32(char_ptr + 0x448);
      frame.characters[4].posY = accessors->ReadF32(char_ptr + 0x44C);
      frame.characters[4].actionState = Memory::Read_U32(char_ptr + 0x1d4);
      frame.characters[4].heading = Memory::Read_U16(char_ptr + 0x42);

      // Read effect type from char_ptr + 0x11C
      u32 effect_ptr = Memory::Read_U32(char_ptr + 0x11C);
      if (effect_ptr == Metadata::addressEffectFrozen)
        frame.characters[4].effectType = 1;
      else if (effect_ptr == Metadata::addressEffectOnFire)
        frame.characters[4].effectType = 2;
      else if (effect_ptr == Metadata::addressEffectStar)
        frame.characters[4].effectType = 3;
      else if (effect_ptr == Metadata::addressEffectElectrocuted)
        frame.characters[4].effectType = 4;
      else
        frame.characters[4].effectType = 0;  // No effect

      // Read speed item state from char_ptr + 0x370, 0x374, 0x36c
      s32 speed_item_type = static_cast<s32>(Memory::Read_U32(char_ptr + 0x370));
      frame.characters[4].speedItemType = (speed_item_type == -1) ? 0 : static_cast<u8>(speed_item_type);
      frame.characters[4].speedItemCount = static_cast<u8>(Memory::Read_U32(char_ptr + 0x374));
      frame.characters[4].speedItemTimer = accessors->ReadF32(char_ptr + 0x36c);
    }
  }

  // Right strikers (indices 5-8)
  for (int i = 0; i < 4; i++)
  {
    u32 char_ptr = Memory::Read_U32(Metadata::addressCharacterPointersBase + 0x10 + (i * 4));
    if (char_ptr != 0)
    {
      frame.characters[5 + i].posX = accessors->ReadF32(char_ptr + 0x520);
      frame.characters[5 + i].posY = accessors->ReadF32(char_ptr + 0x524);
      frame.characters[5 + i].actionState = Memory::Read_U32(char_ptr + 0x1d8);
      frame.characters[5 + i].heading = Memory::Read_U16(char_ptr + 0x42);

      // Read effect type from char_ptr + 0x11C
      u32 effect_ptr = Memory::Read_U32(char_ptr + 0x11C);
      if (effect_ptr == Metadata::addressEffectFrozen)
        frame.characters[5 + i].effectType = 1;
      else if (effect_ptr == Metadata::addressEffectOnFire)
        frame.characters[5 + i].effectType = 2;
      else if (effect_ptr == Metadata::addressEffectStar)
        frame.characters[5 + i].effectType = 3;
      else if (effect_ptr == Metadata::addressEffectElectrocuted)
        frame.characters[5 + i].effectType = 4;
      else
        frame.characters[5 + i].effectType = 0;  // No effect

      // Read speed item state from char_ptr + 0x370, 0x374, 0x36c
      s32 speed_item_type = static_cast<s32>(Memory::Read_U32(char_ptr + 0x370));
      frame.characters[5 + i].speedItemType = (speed_item_type == -1) ? 0 : static_cast<u8>(speed_item_type);
      frame.characters[5 + i].speedItemCount = static_cast<u8>(Memory::Read_U32(char_ptr + 0x374));
      frame.characters[5 + i].speedItemTimer = accessors->ReadF32(char_ptr + 0x36c);
    }
  }

  // Right goalie (index 9)
  {
    u32 char_ptr = Memory::Read_U32(Metadata::addressRightGoaliePointer);
    if (char_ptr != 0)
    {
      frame.characters[9].posX = accessors->ReadF32(char_ptr + 0x448);
      frame.characters[9].posY = accessors->ReadF32(char_ptr + 0x44C);
      frame.characters[9].actionState = Memory::Read_U32(char_ptr + 0x1d4);
      frame.characters[9].heading = Memory::Read_U16(char_ptr + 0x42);

      // Read effect type from char_ptr + 0x11C
      u32 effect_ptr = Memory::Read_U32(char_ptr + 0x11C);
      if (effect_ptr == Metadata::addressEffectFrozen)
        frame.characters[9].effectType = 1;
      else if (effect_ptr == Metadata::addressEffectOnFire)
        frame.characters[9].effectType = 2;
      else if (effect_ptr == Metadata::addressEffectStar)
        frame.characters[9].effectType = 3;
      else if (effect_ptr == Metadata::addressEffectElectrocuted)
        frame.characters[9].effectType = 4;
      else
        frame.characters[9].effectType = 0;  // No effect

      // Read speed item state from char_ptr + 0x370, 0x374, 0x36c
      s32 speed_item_type = static_cast<s32>(Memory::Read_U32(char_ptr + 0x370));
      frame.characters[9].speedItemType = (speed_item_type == -1) ? 0 : static_cast<u8>(speed_item_type);
      frame.characters[9].speedItemCount = static_cast<u8>(Memory::Read_U32(char_ptr + 0x374));
      frame.characters[9].speedItemTimer = accessors->ReadF32(char_ptr + 0x36c);
    }
  }
}

void GameStateCapture::ReadControllerInputs(GameStateFrame& frame)
{
  // Read controller inputs from SI callback buffer (s_buffered_inputs)
  // These are the EXACT values from Movie::PlayController / RecordInput,
  // captured at the SI device layer with no timing ambiguity.
  for (int port = 0; port < GameStateFrame::CONTROLLER_COUNT; port++)
  {
    if (!s_buffered_inputs[port].valid)
    {
      frame.controllers[port] = {};
      continue;
    }

    const GCPadStatus& pad = s_buffered_inputs[port].pad;
    frame.controllers[port].buttons = pad.button;
    frame.controllers[port].stickX = pad.stickX;
    frame.controllers[port].stickY = pad.stickY;
    frame.controllers[port].substickX = pad.substickX;
    frame.controllers[port].substickY = pad.substickY;
    frame.controllers[port].triggerLeft = pad.triggerLeft;
    frame.controllers[port].triggerRight = pad.triggerRight;
    frame.controllers[port].isConnected = pad.isConnected ? 1 : 0;
  }

  // Debug logging
  static int debug_frame = 0;
  if (debug_frame < 3)
  {
    INFO_LOG_FMT(CORE, "CITF Frame {} (inputCount {}) Port 0: button=0x{:04X} stick=({},{}) cstick=({},{}) L={} R={} connected={}",
                 debug_frame, frame.movieFrameNumber,
                 frame.controllers[0].buttons,
                 frame.controllers[0].stickX, frame.controllers[0].stickY,
                 frame.controllers[0].substickX, frame.controllers[0].substickY,
                 frame.controllers[0].triggerLeft, frame.controllers[0].triggerRight,
                 frame.controllers[0].isConnected);
    debug_frame++;
  }
}

void GameStateCapture::ReadPowerupInventory(GameStateFrame& frame)
{
  // Read team powerup inventory slots from cTeam objects
  // Each team has 2 slots at offsets +0x44 and +0x50
  // Each slot: s32 type (at +0x0), u32 chargeCount (at +0x4), u8 isNew (at +0x8)

  // Left team (team 1)
  u32 team1_ptr = Memory::Read_U32(Metadata::addressTeam1Pointer);
  if (team1_ptr != 0)
  {
    // Slot 0
    frame.leftTeamInventory[0].type = static_cast<s32>(Memory::Read_U32(team1_ptr + 0x44));
    frame.leftTeamInventory[0].chargeCount = static_cast<u8>(Memory::Read_U32(team1_ptr + 0x48));
    frame.leftTeamInventory[0].isNew = Memory::Read_U8(team1_ptr + 0x4C);

    // Slot 1
    frame.leftTeamInventory[1].type = static_cast<s32>(Memory::Read_U32(team1_ptr + 0x50));
    frame.leftTeamInventory[1].chargeCount = static_cast<u8>(Memory::Read_U32(team1_ptr + 0x54));
    frame.leftTeamInventory[1].isNew = Memory::Read_U8(team1_ptr + 0x58);
  }

  // Right team (team 2)
  u32 team2_ptr = Memory::Read_U32(Metadata::addressTeam2Pointer);
  if (team2_ptr != 0)
  {
    // Slot 0
    frame.rightTeamInventory[0].type = static_cast<s32>(Memory::Read_U32(team2_ptr + 0x44));
    frame.rightTeamInventory[0].chargeCount = static_cast<u8>(Memory::Read_U32(team2_ptr + 0x48));
    frame.rightTeamInventory[0].isNew = Memory::Read_U8(team2_ptr + 0x4C);

    // Slot 1
    frame.rightTeamInventory[1].type = static_cast<s32>(Memory::Read_U32(team2_ptr + 0x50));
    frame.rightTeamInventory[1].chargeCount = static_cast<u8>(Memory::Read_U32(team2_ptr + 0x54));
    frame.rightTeamInventory[1].isNew = Memory::Read_U8(team2_ptr + 0x58);
  }
}

void GameStateCapture::ReadItems(GameStateFrame& frame)
{
  const AddressSpace::Accessors* accessors =
      AddressSpace::GetAccessors(AddressSpace::Type::Effective);

  // Read active powerups from the 25-slot array at addressActivePowerupArray
  // Each slot contains a pointer to a PowerupBase object (or NULL if inactive)
  frame.itemCount = 0;

  for (int slot = 0; slot < 25 && frame.itemCount < GameStateFrame::MAX_ITEMS; slot++)
  {
    u32 powerup_ptr = Memory::Read_U32(Metadata::addressActivePowerupArray + (slot * 4));

    if (powerup_ptr == 0)
      continue;  // Empty slot

    FrameItem& item = frame.items[frame.itemCount];

    // Position (3x float at +0x2C, +0x30, +0x34)
    item.posX = accessors->ReadF32(powerup_ptr + 0x2C);
    item.posY = accessors->ReadF32(powerup_ptr + 0x30);
    item.posZ = accessors->ReadF32(powerup_ptr + 0x34);

    // Velocity (3x float at +0x44, +0x48, +0x4C)
    item.velX = accessors->ReadF32(powerup_ptr + 0x44);
    item.velY = accessors->ReadF32(powerup_ptr + 0x48);
    item.velZ = accessors->ReadF32(powerup_ptr + 0x4C);

    // Powerup type (u32 at +0x18, but we store as u8)
    item.powerupType = static_cast<u8>(Memory::Read_U32(powerup_ptr + 0x18));

    // Strength level (u32 at +0x6C, but we store as u8: 0=weak, 1=medium, 2=strong)
    item.strengthLevel = static_cast<u8>(Memory::Read_U32(powerup_ptr + 0x6C));

    // Slot index (u32 at +0x64)
    item.slotIndex = static_cast<u8>(Memory::Read_U32(powerup_ptr + 0x64));

    // Thrower pointer (u32 at +0x14)
    item.throwerPointer = Memory::Read_U32(powerup_ptr + 0x14);

    // Lifetime timer (u32 at +0x1C)
    item.lifetimeTimer = Memory::Read_U32(powerup_ptr + 0x1C);

    // Target team pointer (u32 at +0x10)
    item.targetTeamPointer = Memory::Read_U32(powerup_ptr + 0x10);

    // Speed multiplier (float at +0x28)
    item.speedMultiplier = accessors->ReadF32(powerup_ptr + 0x28);

    // Random ID (u16 at +0x24)
    item.randomID = Memory::Read_U16(powerup_ptr + 0x24);

    frame.itemCount++;
  }
}

void GameStateCapture::EndCapture(const std::string& output_path)
{
  if (!s_capturing)
    return;

  s_capturing = false;

  if (s_frame_buffer.empty())
  {
    INFO_LOG_FMT(CORE, "GameStateCapture: No frames captured, skipping file write");
    return;
  }

  File::IOFile file(output_path, "wb");
  if (!file.IsOpen())
  {
    ERROR_LOG_FMT(CORE, "GameStateCapture: Failed to open output file: {}", output_path);
    s_frame_buffer.clear();
    return;
  }

  // Fixed portion size: everything before the items array
  constexpr u32 fixed_frame_size = static_cast<u32>(offsetof(GameStateFrame, items));

  // Build and write header
  CaptureHeader header;
  std::memset(&header, 0, sizeof(header));
  header.magic[0] = 'C';
  header.magic[1] = 'I';
  header.magic[2] = 'T';
  header.magic[3] = 'F';
  header.version = CITF_VERSION;
  header.frameCount = static_cast<u32>(s_frame_buffer.size());
  header.fixedFrameSize = fixed_frame_size;
  header.leftCaptainID = static_cast<u8>(Metadata::getLeftSideCaptainID());
  header.rightCaptainID = static_cast<u8>(Metadata::getRightSideCaptainID());
  header.leftSidekickID = static_cast<u8>(Metadata::getLeftSideSidekickID());
  header.rightSidekickID = static_cast<u8>(Metadata::getRightSideSidekickID());
  header.stadiumID = static_cast<u8>(Metadata::getStadiumID());

  file.WriteBytes(&header, sizeof(header));

  // Write variable-length frames: fixed portion + only active items
  size_t total_bytes = sizeof(header);
  for (const auto& frame : s_frame_buffer)
  {
    file.WriteBytes(&frame, fixed_frame_size);
    if (frame.itemCount > 0)
    {
      file.WriteBytes(frame.items, frame.itemCount * sizeof(FrameItem));
    }
    total_bytes += fixed_frame_size + frame.itemCount * sizeof(FrameItem);
  }

  INFO_LOG_FMT(CORE, "GameStateCapture: Wrote {} frames ({} bytes) to {}", header.frameCount,
               total_bytes, output_path);

  s_frame_buffer.clear();
}

#include "GameStateFrame.h"

#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>
#include <zstd.h>

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
static CaptureMatchInfo s_match_info;

// Buffered controller inputs from last Movie::PlayController() call
struct BufferedInput
{
  GCPadStatus pad;
  u64 inputCount;
  bool valid;
};
static std::array<BufferedInput, 4> s_buffered_inputs{};

// Binary file format constants
static constexpr u32 CITF_VERSION = 11;

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

void GameStateCapture::SetMatchInfo(const CaptureMatchInfo& info)
{
  s_match_info = info;
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

  // Game phase and potential scorer from cGame singleton
  constexpr u32 cGameSingletonPtr = 0x80373708;
  u32 cGamePtr = Memory::Read_U32(cGameSingletonPtr);
  if (cGamePtr != 0)
  {
    // eGameState enum at cGame+0x24: 0=pre-match, 1=kickoff, 2=goal celebration, 3=transition,
    // 4=active play, 5=active play variant
    frame.gamePhase = static_cast<u8>(Memory::Read_U32(cGamePtr + 0x24));

    // Potential scorer at cGame+0x2C: set on ball pickup (cBall::SetOwner) and on shot fire
    // (zz_80020164_ for action states 0x05/0x08/0x11). Persists through ball-in-flight, so
    // perfect-pass goals correctly credit the shooter rather than showing owner=NONE.
    frame.potentialScorerPtr = Memory::Read_U32(cGamePtr + 0x2C);
  }

  // Ball, characters, controllers, powerup inventory, items
  ReadBallState(frame);
  ReadCharacterState(frame);
  ReadControllerInputs(frame);
  ReadPowerupInventory(frame);
  ReadTeamStats(frame);
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

  // Ball velocity, ownership, and perfect pass state — all from the cBall object.
  // cBall::SetVelocity writes to ball_ptr+0x58/5C/60 and PostPhysicsUpdate refreshes them
  // each frame from the physics engine. Static address mirrors lag 1-2 frames behind.
  u32 ball_ptr = Memory::Read_U32(Metadata::addressBallPointer);
  if (ball_ptr != 0)
  {
    frame.ballVelX = accessors->ReadF32(ball_ptr + 0x58);
    frame.ballVelY = accessors->ReadF32(ball_ptr + 0x5C);
    frame.ballVelZ = accessors->ReadF32(ball_ptr + 0x60);
    frame.ballOwnerCharacterPointer = Memory::Read_U32(ball_ptr + 0x24);
    frame.ballPassTargetPointer = Memory::Read_U32(ball_ptr + 0x30);
    frame.isPerfectPass = Memory::Read_U8(ball_ptr + 0xa1);
  }

  // Ball charge level (shared by both teams)
  frame.ballChargeAmount =
      static_cast<float>(Memory::Read_U32(Metadata::addressChargedBallAmount));
}

void GameStateCapture::ReadCharacterState(GameStateFrame& frame)
{
  const AddressSpace::Accessors* accessors =
      AddressSpace::GetAccessors(AddressSpace::Type::Effective);

  // Left strikers (indices 0-3)
  for (int i = 0; i < 4; i++)
  {
    u32 char_ptr = Memory::Read_U32(Metadata::addressCharacterPointersBase + (i * 4));
    frame.characterPointers[i] = char_ptr;
    if (char_ptr != 0)
    {
      // Base entity position: +0x18=X, +0x1c=Y, +0x20=Z (used by game logic in SwapController,
      // InControlOfBall, etc. — same coordinate system as ball)
      frame.characters[i].posX = accessors->ReadF32(char_ptr + 0x18);
      frame.characters[i].posY = accessors->ReadF32(char_ptr + 0x1c);
      frame.characters[i].posZ = accessors->ReadF32(char_ptr + 0x20);
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

      // Human-controlled flag: ptr+0x1c0 is controller pointer (non-null = human)
      frame.characters[i].isUserControlled = (Memory::Read_U32(char_ptr + 0x1c0) != 0) ? 1 : 0;
    }
  }

  // Left goalie (index 4)
  {
    u32 char_ptr = Memory::Read_U32(Metadata::addressLeftGoaliePointer);
    frame.characterPointers[4] = char_ptr;
    if (char_ptr != 0)
    {
      frame.characters[4].posX = accessors->ReadF32(char_ptr + 0x18);
      frame.characters[4].posY = accessors->ReadF32(char_ptr + 0x1c);
      frame.characters[4].posZ = accessors->ReadF32(char_ptr + 0x20);
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

      // Human-controlled flag: ptr+0x1c0 is controller pointer (non-null = human)
      frame.characters[4].isUserControlled = (Memory::Read_U32(char_ptr + 0x1c0) != 0) ? 1 : 0;
    }
  }

  // Right strikers (indices 5-8)
  for (int i = 0; i < 4; i++)
  {
    u32 char_ptr = Memory::Read_U32(Metadata::addressCharacterPointersBase + 0x10 + (i * 4));
    frame.characterPointers[5 + i] = char_ptr;
    if (char_ptr != 0)
    {
      frame.characters[5 + i].posX = accessors->ReadF32(char_ptr + 0x18);
      frame.characters[5 + i].posY = accessors->ReadF32(char_ptr + 0x1c);
      frame.characters[5 + i].posZ = accessors->ReadF32(char_ptr + 0x20);
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

      // Human-controlled flag: ptr+0x1c0 is controller pointer (non-null = human)
      frame.characters[5 + i].isUserControlled = (Memory::Read_U32(char_ptr + 0x1c0) != 0) ? 1 : 0;
    }
  }

  // Right goalie (index 9)
  {
    u32 char_ptr = Memory::Read_U32(Metadata::addressRightGoaliePointer);
    frame.characterPointers[9] = char_ptr;
    if (char_ptr != 0)
    {
      frame.characters[9].posX = accessors->ReadF32(char_ptr + 0x18);
      frame.characters[9].posY = accessors->ReadF32(char_ptr + 0x1c);
      frame.characters[9].posZ = accessors->ReadF32(char_ptr + 0x20);
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

      // Human-controlled flag: ptr+0x1c0 is controller pointer (non-null = human)
      frame.characters[9].isUserControlled = (Memory::Read_U32(char_ptr + 0x1c0) != 0) ? 1 : 0;
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

void GameStateCapture::ReadTeamStats(GameStateFrame& frame)
{
  // Left team stats
  frame.leftStats.shots = static_cast<u16>(Memory::Read_U32(Metadata::addressLeftSideShots));
  frame.leftStats.hits = Memory::Read_U16(Metadata::addressLeftSideHits);
  frame.leftStats.steals = Memory::Read_U16(Metadata::addressLeftSideSteals);
  frame.leftStats.superStrikes = Memory::Read_U16(Metadata::addressLeftSideSuperStrikes);
  frame.leftStats.perfectPasses = Memory::Read_U16(Metadata::addressLeftSidePerfectPasses);

  // Right team stats
  frame.rightStats.shots = static_cast<u16>(Memory::Read_U32(Metadata::addressRightSideShots));
  frame.rightStats.hits = Memory::Read_U16(Metadata::addressRightSideHits);
  frame.rightStats.steals = Memory::Read_U16(Metadata::addressRightSideSteals);
  frame.rightStats.superStrikes = Memory::Read_U16(Metadata::addressRightSideSuperStrikes);
  frame.rightStats.perfectPasses = Memory::Read_U16(Metadata::addressRightSidePerfectPasses);
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

  // Fixed portion size: everything before the items array
  constexpr u32 fixed_frame_size = static_cast<u32>(offsetof(GameStateFrame, items));

  // Build header
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

  // Field geometry (static per game — read from known data addresses)
  const AddressSpace::Accessors* accessors =
      AddressSpace::GetAccessors(AddressSpace::Type::Effective);
  header.goalLineX = accessors->ReadF32(0x802a3e60);
  header.sidelineY = accessors->ReadF32(0x802a3e64);
  header.penaltyBoxX = accessors->ReadF32(0x80371108);
  header.netHalfWidth = accessors->ReadF32(0x80371204);
  header.netHeight = accessors->ReadF32(0x80371200);
  header.netDepth = accessors->ReadF32(0x8037120c);

  // v11 match metadata — sourced from s_match_info (populated by SetMatchInfo from CIT JSON)
  header.epoch                = s_match_info.epoch;
  header.citrusGameId         = s_match_info.citrusGameId;
  header.submittedByDiscordId = s_match_info.submittedByDiscordId;
  header.roomId               = s_match_info.roomId;
  header.gameCount            = s_match_info.gameCount;
  header.isRanked             = s_match_info.isRanked ? 1 : 0;
  header.isNetplay            = s_match_info.isNetplay ? 1 : 0;
  header.matchTimeAllotted    = s_match_info.matchTimeAllotted;
  header.matchDifficulty      = s_match_info.matchDifficulty;
  header.matchItems           = s_match_info.matchItems ? 1 : 0;
  header.matchSuperStrikes    = s_match_info.matchSuperStrikes ? 1 : 0;
  header.matchBowserOrFTX     = s_match_info.matchBowserOrFTX ? 1 : 0;
  header.overtimeNotReached   = s_match_info.overtimeNotReached ? 1 : 0;
  header.matchTimeElapsed     = s_match_info.matchTimeElapsed;
  std::memcpy(header.md5, s_match_info.md5.data(), 16);
  for (int i = 0; i < 4; i++)
  {
    const auto& pe = s_match_info.ports[i];
    header.portTeam[i]                = pe.team;
    header.portPlayers[i].discordId   = pe.discordId;
    std::strncpy(header.portPlayers[i].displayName, pe.displayName.c_str(), 31);
    header.portPlayers[i].displayName[31] = '\0';
  }

  // Assemble raw binary: header then variable-length frames
  size_t raw_size = sizeof(header);
  for (const auto& frame : s_frame_buffer)
    raw_size += fixed_frame_size + frame.itemCount * sizeof(FrameItem);

  std::vector<u8> raw;
  raw.reserve(raw_size);

  const u8* hdr_bytes = reinterpret_cast<const u8*>(&header);
  raw.insert(raw.end(), hdr_bytes, hdr_bytes + sizeof(header));

  for (const auto& frame : s_frame_buffer)
  {
    const u8* frame_bytes = reinterpret_cast<const u8*>(&frame);
    raw.insert(raw.end(), frame_bytes, frame_bytes + fixed_frame_size);
    if (frame.itemCount > 0)
    {
      const u8* item_bytes = reinterpret_cast<const u8*>(frame.items);
      raw.insert(raw.end(), item_bytes, item_bytes + frame.itemCount * sizeof(FrameItem));
    }
  }

  s_frame_buffer.clear();

  // Compress and write on a background thread — zstd level 19 can take 1-2s on a full match
  u32 frame_count = header.frameCount;
  std::thread([raw = std::move(raw), output_path, frame_count]() mutable {
    const size_t compress_bound = ZSTD_compressBound(raw.size());
    std::vector<u8> compressed(compress_bound);
    const size_t compressed_size =
        ZSTD_compress(compressed.data(), compress_bound, raw.data(), raw.size(), 19);

    if (ZSTD_isError(compressed_size))
    {
      ERROR_LOG_FMT(CORE, "GameStateCapture: zstd compression failed: {}",
                    ZSTD_getErrorName(compressed_size));
      return;
    }

    File::IOFile file(output_path, "wb");
    if (!file.IsOpen())
    {
      ERROR_LOG_FMT(CORE, "GameStateCapture: Failed to open output file: {}", output_path);
      return;
    }

    file.WriteBytes(compressed.data(), compressed_size);

    INFO_LOG_FMT(CORE, "GameStateCapture: Wrote {} frames ({} bytes raw -> {} bytes compressed, "
                 "{:.1f}x ratio) to {}",
                 frame_count, raw.size(), compressed_size,
                 static_cast<float>(raw.size()) / compressed_size, output_path);
  }).detach();
}

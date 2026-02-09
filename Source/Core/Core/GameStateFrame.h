#pragma once

#include <string>
#include <vector>

#include "Common/CommonTypes.h"

struct GCPadStatus;

#pragma pack(push, 1)

// Written once in the file header — static match info
struct CaptureHeader
{
  char magic[4];          // "CITF"
  u32 version;            // 4
  u32 frameCount;
  u32 fixedFrameSize;     // size of fixed portion per frame (excludes variable items)

  // Match info (static for entire replay)
  u8 leftCaptainID;       // captain enum (0=Daisy,1=DK,2=Luigi,3=Mario,4=Peach,5=Waluigi,6=Wario,7=Yoshi,8=SuperTeam)
  u8 rightCaptainID;
  u8 leftSidekickID;      // sidekick enum (0=Toad,1=Koopa,2=HammerBro,3=Birdo,8=SuperTeam)
  u8 rightSidekickID;
  u8 stadiumID;           // 0=Pipeline,1=Palace,2=Konga,3=Underground,4=Crater,5=Bowser,6=BattleDome
  u8 headerPadding[3];
};
// 24 bytes

struct FrameControllerInput
{
  u16 buttons;      // OR'd PAD_BUTTON_* flags
  u8 stickX;        // 0-255 (0x80 = center)
  u8 stickY;
  u8 substickX;     // C-stick
  u8 substickY;
  u8 triggerLeft;   // 0-255
  u8 triggerRight;
  u8 isConnected;   // 1 = yes, 0 = no
  u8 padding;
};
// 10 bytes per controller

struct FramePowerupInventorySlot
{
  s32 type;         // ePowerUpType enum (-1=empty, 0=Green Shell, 1=Red Shell, 2=Shell, 3=Blue Shell, 4=Banana, 5=Bob-omb)
  u8 chargeCount;   // Number of items spawned on use (1, 3, or 5)
  u8 isNew;         // 1 = freshly awarded, 0 = already seen
  u8 padding[2];
};
// 8 bytes per inventory slot (aligned to 4-byte boundary)

struct FrameCharacter
{
  float posX;
  float posY;
  u32 actionState;     // striker: ptr+0x1d8, goalie: ptr+0x1d4 (0x0-0x1b, 0xFF=clearing)
  u16 heading;         // 16-bit fixed-point angle (0x0000-0xFFFF = 0-360°, from ptr+0x42)
  u8 effectType;       // Effect enum from ptr+0x11C: 0=none, 1=frozen, 2=on fire, 3=star, 4=electrocuted
  u8 speedItemType;    // Speed item from ptr+0x370: 0=none, 7=mushroom, 8=star (-1 stored as 0)
  u8 speedItemCount;   // Item count from ptr+0x374: 1, 3, or 5 (0 if none active)
  u8 padding[3];       // Align to 4-byte boundary
  float speedItemTimer; // Timer from ptr+0x36c: seconds remaining (0.0 if none active)
};
// 24 bytes per character
// Slot layout: [0-3] left strikers, [4] left goalie,
//              [5-8] right strikers, [9] right goalie
// Team/role is implicit from slot index
// Heading: 0°=East, 90°=North, 180°=West, 270°=South (counterclockwise)

struct FrameItem
{
  float posX;               // Position from PowerupBase + 0x2C
  float posY;               // Position from PowerupBase + 0x30
  float posZ;               // Position from PowerupBase + 0x34
  float velX;               // Velocity from PowerupBase + 0x44
  float velY;               // Velocity from PowerupBase + 0x48
  float velZ;               // Velocity from PowerupBase + 0x4C
  u8 powerupType;           // ePowerUpType enum (0=Green Shell, 1=Red Shell, 2=Shell, 3=Blue Shell, 4=Banana, 5=Bob-omb)
  u8 strengthLevel;         // 0=weak, 1=medium, 2=strong (from PowerupBase + 0x6C)
  u8 slotIndex;             // Index in main powerup array 0-24 (from PowerupBase + 0x64)
  u8 padding;
  u32 throwerPointer;       // Pointer to character who threw this (from PowerupBase + 0x14)
  u32 lifetimeTimer;        // Countdown timer in frames (from PowerupBase + 0x1C)
  u32 targetTeamPointer;    // Pointer to target team (from PowerupBase + 0x10)
  float speedMultiplier;    // Speed multiplier (from PowerupBase + 0x28)
  u16 randomID;             // Random ID 0-65000 (from PowerupBase + 0x24)
  u16 padding2;
};
// 48 bytes per powerup

struct GameStateFrame
{
  // -- Frame metadata (8 bytes) --
  // frameIndex is implicit from position in file
  float gameTime;
  u32 movieFrameNumber;  // DTM input index (from Movie::GetCurrentInputCount, accounts for lag frames)

  // -- Score (4 bytes) --
  u8 leftScore;
  u8 rightScore;
  u8 isPaused;
  u8 padding1;

  // -- Ball state (32 bytes) --
  float ballPosX, ballPosY, ballPosZ;
  float ballVelX, ballVelY, ballVelZ;
  u32 ballOwnerCharacterPointer;  // character pointer of who possesses the ball
  u8 isPerfectPass;               // 1 = perfect pass active
  u8 ballPadding[3];

  // -- Characters: 10 total (120 bytes) --
  static constexpr int CHARACTER_COUNT = 10;
  FrameCharacter characters[CHARACTER_COUNT];

  // -- Controller inputs: 4 ports (40 bytes) --
  static constexpr int CONTROLLER_COUNT = 4;
  FrameControllerInput controllers[CONTROLLER_COUNT];

  // -- Powerup inventory: 2 teams x 2 slots (32 bytes) --
  static constexpr int TEAM_COUNT = 2;
  static constexpr int INVENTORY_SLOTS_PER_TEAM = 2;
  FramePowerupInventorySlot leftTeamInventory[INVENTORY_SLOTS_PER_TEAM];   // Left team slots 0 and 1
  FramePowerupInventorySlot rightTeamInventory[INVENTORY_SLOTS_PER_TEAM];  // Right team slots 0 and 1

  // -- Items: max 25 active powerups (variable on disk, fixed in memory) --
  static constexpr int MAX_ITEMS = 25;
  u8 itemCount;
  u8 padding2[3];
  FrameItem items[MAX_ITEMS];
};

#pragma pack(pop)

// Capture system for recording per-frame game state during replay playback.
// Static interface following the Metadata/StateAuxillary pattern.
class GameStateCapture
{
public:
  // Called once when playback match starts
  static void BeginCapture();

  // Called each frame from OnFrameEnd() during match
  static void CaptureFrame();

  // Called when match ends - writes binary file to disk
  static void EndCapture(const std::string& output_path);

  // Check if capture is active
  static bool IsCapturing();

  // Called immediately when controller input is read during playback
  static void OnControllerInput(int port, const GCPadStatus& pad, u64 inputCount);

private:
  static void ReadCharacterState(GameStateFrame& frame);
  static void ReadBallState(GameStateFrame& frame);
  static void ReadControllerInputs(GameStateFrame& frame);
  static void ReadPowerupInventory(GameStateFrame& frame);
  static void ReadItems(GameStateFrame& frame);
};

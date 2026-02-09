#pragma once

#include <string>
#include <map>
#include <Core/NetPlayProto.h>
#include <Core/NetPlayClient.h>

class Metadata
{
public:
  static std::string getJSONString();
  static void writeJSON(std::string jsonString, bool callBatch = true);
  static void setMatchMetadata();
  static void setPlayerName(std::string playerNameParam);
  static void setPlayerArray(std::vector<const NetPlay::Player*>);
  static void setNetPlayControllers(NetPlay::PadMappingArray m_pad_map);
  static void setMD5(std::array<u8, 16> md5Param);
  static std::vector<const NetPlay::Player*> getPlayerArray();
  static NetPlay::PadMappingArray getControllers();
  static void setNetPlayRoomCode(std::string roomIDParam);
  static void setGameCount(int gameCountParam);
  static void setCitrusUser(CitrusUser citrusUserParam);
  static void setMatchDateString();
  static void setCitrusGameId();
  static int getMatchMode();
  static int getLeftSideCaptainID();
  static int getRightSideCaptainID();
  static int getLeftSideSidekickID();
  static int getRightSideSidekickID();
  static int getStadiumID();
  static int getGameCount();
  static CitrusUser getCitrusUser();
  static std::string getCitrusGameId();
  static u64 getCitrusGameIdAsInt();

  // CONSTANTS

  // Ball position (float)
  static const u32 addressBallXPos = 0x80312adc;
  static const u32 addressBallYPos = 0x80312ae0;
  static const u32 addressBallZPos = 0x80312ae4;

  // Ball velocity (float)
  static const u32 addressBallXVel = 0x80311010;
  static const u32 addressBallYVel = 0x80311014;
  static const u32 addressBallZVel = 0x80311018;

  // Controller input base (8 bytes per port: buttons, sticks, triggers)
  static const u32 addressControllerInputBase = 0x8032C348;

  // Ball object pointer (read to get ball pointer, then use offsets)
  // ball_ptr + 0x24 -> u32 character pointer of who owns the ball
  // ball_ptr + 0xa1 -> u8 perfect pass state (1 = perfect pass active)
  static const u32 addressBallPointer = 0x80373664;

  // Team pointers (cTeam objects)
  static const u32 addressTeam1Pointer = 0x80371238;  // Left/team 1
  static const u32 addressTeam2Pointer = 0x8037123C;  // Right/team 2
  // Team inventory offsets from team pointer:
  // +0x44 -> slot 0 (s32 type, u32 chargeCount, u8 isNew)
  // +0x50 -> slot 1 (s32 type, u32 chargeCount, u8 isNew)

  // Active powerup array (25 PowerupBase* pointers)
  static const u32 addressActivePowerupArray = 0x802A76DC;
  // PowerupBase struct offsets:
  // +0x2C -> position (3x float XYZ)
  // +0x44 -> velocity (3x float XYZ)
  // +0x18 -> u32 powerup type (ePowerUpType enum)
  // +0x6C -> u32 strength level (0/1/2)
  // +0x64 -> u32 slot index
  // +0x14 -> u32 thrower (player pointer)
  // +0x1C -> u32 lifetime timer
  // +0x10 -> u32 target team pointer
  // +0x28 -> float speed multiplier
  // +0x24 -> u16 random ID

  // Goalie pointers (derived from addressCharacterPointersBase)
  static const u32 addressLeftGoaliePointer = 0x8030d530;   // base + 0x20
  static const u32 addressRightGoaliePointer = 0x8030d534;  // base + 0x24

  // Character effect type pointers (compare against u32 at char_ptr + 0x11C)
  static const u32 addressEffectFrozen = 0x802af514;
  static const u32 addressEffectOnFire = 0x802af520;
  static const u32 addressEffectStar = 0x802af52c;
  static const u32 addressEffectElectrocuted = 0x802af538;

  static const u32 addressControllerPort1 = 0x81536A04;
  static const u32 addressControllerPort2 = 0x81536A06;
  static const u32 addressControllerPort3 = 0x81536A08;
  static const u32 addressControllerPort4 = 0x81536A0A;

  static const u32 addressLeftSideCaptainID = 0x815369f0;
  static const u32 addressRightSideCaptainID = 0x815369f4;
  static const u32 addressLeftSideSidekickID = 0x815369f8;
  static const u32 addressRightSideSidekickID = 0x815369fc;
  static const u32 addressStadiumID = 0x81536a00;
  static const u32 addressCharacterPointersBase = 0x8030d510;
  static const u32 addressIsGamePaused = 0x803725c1; // 1 if the pause screen is active, 0 all other parts of an active game 

  static const u32 addressLeftSideCupCaptainID = 0x8040000c;
  static const u32 addressLeftSideCupSidekickID = 0x8040000d;
  static const u32 addressRightSideCupCaptainID = 0x8040000e;
  static const u32 addressRightSideCupSidekickID = 0x8040000f;
  static const u32 addressCupStadiumID = 0x80400010;

  static const u32 addressCustomTrainingModeEnabled = 0x80400011;
  static const u32 addressCustomTrainingModePossessionChange = 0x80400012;

  static const u32 addressHockeyModeEnabled = 0x80400013; // 8-bit
  // Attack is either a slide tackle or hit. It might not be a penalty since not all attacks are penalties
  static const u32 addressHockeyModeAttackCharacterId = 0x80400014; // 32-bit
  static const u32 addressHockeyModePenaltyFlag = 0x80400018;  // 8-bit

  //left team
  static const u32 addressLeftSideScore = 0x81536a56;
  static const u32 addressLeftSideShots = 0x81536a52;
  static const u32 addressLeftSideHits = 0x81536a6c;
  static const u32 addressLeftSideSteals = 0x81536a6e;
  static const u32 addressLeftSideSuperStrikes = 0x81536af2;
  static const u32 addressLeftSidePerfectPasses = 0x81536a74;

  //right team
  static const u32 addressRightSideScore = 0x81536a58;
  static const u32 addressRightSideShots = 0x81536a92;
  static const u32 addressRightSideHits = 0x81536aac;
  static const u32 addressRightSideSteals = 0x81536aae;
  static const u32 addressRightSideSuperStrikes = 0x81536B32;
  static const u32 addressRightSidePerfectPasses = 0x81536ab4;

  //ruleset
  /*
  81534c68 is 4:3/16:9
  81534c6c is difficulty
  81534c70 is amount of time for game in hex
  81534c74 is power ups on/off
  81534c75 is super strike on/off
  81534c77 is rumble on/off
  81531d76 is bowser attack on/off (81534c76 also is)
  */
  static const u32 addressMatchDifficulty = 0x81534c6c;
  // using custom time allotted instead. this one is what we see in the hud as opposed to ruleset
  static const u32 addressMatchTimeAllotted = 0x80400008;
  static const u32 addressMatchStart = 0x80400000;
  static const u32 addressMatchEnd = 0x80400001;
  static const u32 addressOvertimeNotReachedBool = 0x80400002;
  // 0 for strikers 101, 1 for grudge, 2 for cups/tournaments
  static const u32 addressMatchMode = 0x80400003;
  static const u32 addressTimeElapsed = 0x80400004;
  static const u32 addressMatchItemsBool = 0x81534c74;
  static const u32 addressMatchSuperStrikesBool = 0x81534c75;
  // note, this is same address for first to 7 so need to know if we're on citrus via hash
  static const u32 addressMatchBowserBool = 0x81534c76;

  //stats for item use

  /*
  one byte for item type, one byte for item amount, two bytes filler
  full word (4 bytes) for time
  80410000 for left team item use (start)
  80420000 for right team item use (start)
  80430000 for left team item offset
  80430004 for left team item flag
  80430008 for left team item count
  80430010 for right team item offset
  80430014 for right team item flag
  80430018 for right team item count
  */
  // left team item
  static const u32 addressLeftTeamItemStart = 0x80410000;
  static const u32 addressLeftTeamItemOffset = 0x80430000;
  static const u32 addressLeftTeamGoalOffset = 0x80430004;
  static const u32 addressLeftTeamItemCount = 0x80430008;
  // left team shots
  static const u32 addressLeftTeamMissedShotsOffset = 0x8043000c;
  static const u32 addressLeftTeamMissedShotsFlag = 0x8043000e;
  static const u32 addressLeftTeamMissedShotsBallXPos = 0x80430020;
  static const u32 addressLeftTeamMissedShotsBallYPos = 0x80430024;
  static const u32 addressLeftTeamMissedShotsTimestamp = 0x80430028;
  // to add extra info like ball x/y/z velocity and ball z position
  static const u32 addressLeftTeamMissedShotsVelocityFlag = 0x8043003c;
  static const u32 addressLeftTeamMissedShotsBallZPos = 0x80430040;
  static const u32 addressLeftTeamMissedShotsBallXVel = 0x80430044;
  static const u32 addressLeftTeamMissedShotsBallYVel = 0x80430048;
  static const u32 addressLeftTeamMissedShotsBallZVel = 0x8043004c;
  static const u32 addressLeftTeamMissedShotsBallChargeAmount = 0x80430060;

  // right team item
  static const u32 addressRightTeamItemStart = 0x80420000;
  static const u32 addressRightTeamItemOffset = 0x80430010;
  static const u32 addressRightTeamGoalOffset = 0x80430014;
  static const u32 addressRightTeamItemCount = 0x80430018;
  // right team shots
  static const u32 addressRightTeamMissedShotsOffset = 0x8043001c;
  static const u32 addressRightTeamMissedShotsFlag = 0x8043001e;
  static const u32 addressRightTeamMissedShotsBallXPos = 0x80430030;
  static const u32 addressRightTeamMissedShotsBallYPos = 0x80430034;
  static const u32 addressRightTeamMissedShotsTimestamp = 0x80430038;
  // to add extra info like ball x/y/z velocity and ball z position
  static const u32 addressRightTeamMissedShotsVelocityFlag = 0x8043003e;
  static const u32 addressRightTeamMissedShotsBallZPos = 0x80430050;
  static const u32 addressRightTeamMissedShotsBallXVel = 0x80430054;
  static const u32 addressRightTeamMissedShotsBallYVel = 0x80430058;
  static const u32 addressRightTeamMissedShotsBallZVel = 0x8043005c;
  static const u32 addressRightTeamMissedShotsBallChargeAmount = 0x80430064;

  // charged amount -- shared by left and right
  static const u32 addressChargedBallAmount = 0x80430068;

  // left team goal
  static const u32 addressLeftTeamGoalStart = 0x80440000;
  // right team goal
  static const u32 addressRightTeamGoalStart = 0x80450000;

  // left team missed shots
  static const u32 addressLeftTeamMissedShotsStart = 0x80460000;
  // right team missed shots
  static const u32 addressRightTeamMissedShotsStart = 0x80470000;

  // i thought about putting these next 2 at 8040000c/e and making them half words
  // but if a player got more than ffff, aka 18 minutes, we would experience disparity
  // 
  // left team ball possessed frames
  static const u32 addressLeftTeamBallOwnedFrames = 0x80480000;
  // right team ball possessed frames
  static const u32 addressRightTeamBallOwnedFrames = 0x80480004;
};

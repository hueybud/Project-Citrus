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
  static void setMatchMetadata(tm* matchDateTimeParam);
  static void setPlayerName(std::string playerNameParam);
  static void setPlayerArray(std::vector<const NetPlay::Player*>);
  static void setNetPlayControllers(NetPlay::PadMappingArray m_pad_map);
  static void setMD5(std::array<u8, 16> md5Param);

  // CONSTANTS

  static const u32 addressControllerPort1 = 0x81536A04;
  static const u32 addressControllerPort2 = 0x81536A06;
  static const u32 addressControllerPort3 = 0x81536A08;
  static const u32 addressControllerPort4 = 0x81536A0A;

  static const u32 addressLeftSideCaptainID = 0x815369f0;
  static const u32 addressRightSideCaptainID = 0x815369f4;
  static const u32 addressLeftSideSidekickID = 0x815369f8;
  static const u32 addressRightSideSidekickID = 0x815369fc;
  static const u32 addressStadiumID = 0x81536a00;

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
  static const u32 addressOvertimeNotReachedBool = 0x80400002;
  // 0x80400003 is if grudge or not
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

  // right team item
  static const u32 addressRightTeamItemStart = 0x80420000;
  static const u32 addressRightTeamItemOffset = 0x80430010;
  static const u32 addressRightTeamGoalOffset = 0x80430014;
  static const u32 addressRightTeamItemCount = 0x80430018;

  // left team goal
  static const u32 addressLeftTeamGoalStart = 0x80440000;
  // right team goal
  static const u32 addressRightTeamGoalStart = 0x80450000;
};

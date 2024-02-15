#pragma once

#include <string>
#include <Core/NetPlayProto.h>
#include "Core/HW/SI/SI_Device.h"
#include "Core/ConfigManager.h"

class StateAuxillary
{
public:
  struct HockeyCharacterInfo
  {
    bool currentlyPenalized;
    float currentlyPenalizedTimeRemaining;
    u32 penaltyXAddress;
    u32 penaltyYAddress;
  };
  static void saveState(const std::string& filename, bool wait = false);
  static void saveStateToTrainingBuffer();
  static void saveStateToTrainingBuffer2();
  static void loadStateFromTrainingBuffer();
  static void loadStateFromTrainingBuffer2();
  static void startRecording();
  static void stopRecording(const std::string replay_path);
  static void endPlayback();
  static void setNetPlayControllers(NetPlay::PadMappingArray m_pad_map, NetPlay::PlayerId m_pid);
  static std::vector<int> getOurNetPlayPorts();
  static bool isSpectator();
  static void setPrePort(SerialInterface::SIDevices currentPort0,
                         SerialInterface::SIDevices currentPort1,
                         SerialInterface::SIDevices currentPort2,
                         SerialInterface::SIDevices currentPort3);
  static void setPostPort();
  static bool getBoolMatchStart();
  static bool getBoolMatchEnd();
  static bool getBoolWroteCodes();
  static bool getOverwriteHomeCaptainPositionTrainingMode();
  static bool getCustomTrainingModeStart();
  static bool getIsRanked();
  static std::map<u32, HockeyCharacterInfo> getHockeyLeftTeamCharacterInfo();
  static std::map<u32, HockeyCharacterInfo> getHockeyRightTeamCharacterInfo();
  static int getHockeyLeftTeamTotalPenalties();
  static int getHockeyRightTeamTotalPenalties();
  static void setBoolMatchStart(bool boolValue);
  static void setBoolMatchEnd(bool boolValue);
  static void setBoolWroteCodes(bool boolValue);
  static void setMatchPlayerNames();
  static void setOverwriteHomeCaptainPositionTrainingMode(bool boolValue);
  static void setCustomTrainingModeStart(bool boolValue);
  static void setIsRanked(bool boolValue);
  static void setHockeyLeftTeamCharacterInfo(u32 characterPointer, HockeyCharacterInfo characterInfo);
  static void setHockeyRightTeamCharacterInfo(u32 characterPointer, HockeyCharacterInfo characterInfo);
  static void hockeyModeInit();
  static void setHockeyLeftTeamTotalPenalties(int hitParam);
  static void setHockeyRightTeamTotalPenalties(int hitParam);
  static void updateHockeyDisplay();
};

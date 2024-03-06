#include "StateAuxillary.h"
#include "Core/State.h"
#include <thread>
#include <Core/HW/SI/SI_Device.h>
#include <Core/Config/MainSettings.h>
#include <Core/Config/WiimoteSettings.h>
#include <Core/HW/Wiimote.h>
#include <Core/Core.h>
#include <Core/Metadata.h>
#include "Common/CommonPaths.h"
#include "Common/FileUtil.h"
#include "HW/Memmap.h"
#include "Movie.h"
#include <VideoCommon/OnScreenDisplay.h>

static bool boolMatchStart = false;
static bool boolMatchEnd = false;
static bool wroteCodes = false;
static bool overwroteHomeCaptainPosTraining = false;
static bool customTrainingModeStart = false;
static std::vector<u8> training_mode_buffer;
static bool isRanked = false;

// the amount of penalties that can be incurred before we start placing players in a power play
static int minPowerPlayPenaltyAmount = 1;
static std::map<u32, StateAuxillary::HockeyCharacterInfo> hockeyLeftTeamCharacterInfo;
static int hockeyLeftTeamTotalPenalties = 0;
static std::map<u32, StateAuxillary::HockeyCharacterInfo> hockeyRightTeamCharacterInfo;
static int hockeyRightTeamTotalPenalties = 0;
static int currentTotalScore = 0;

static NetPlay::PadMappingArray netplayGCMap;
// even if the player is using multiple netplay ports to play, say 1 and 3, the game only needs the first one to do proper playback
// therefore, we can use a single int instead of an array
static int ourNetPlayPort;
static std::vector<int> ourNetPlayPortsVector(4);
static SerialInterface::SIDevices preMoviePort0;
static SerialInterface::SIDevices preMoviePort1;
static SerialInterface::SIDevices preMoviePort2;
static SerialInterface::SIDevices preMoviePort3;

bool StateAuxillary::getBoolMatchStart()
{
  return boolMatchStart;
}

bool StateAuxillary::getBoolMatchEnd()
{
  return boolMatchEnd;
}

bool StateAuxillary::getBoolWroteCodes()
{
  return wroteCodes;
}

void StateAuxillary::setBoolMatchStart(bool boolValue)
{
  boolMatchStart = boolValue;
}

void StateAuxillary::setBoolMatchEnd(bool boolValue)
{
  boolMatchEnd = boolValue;
}

void StateAuxillary::setBoolWroteCodes(bool boolValue)
{
  wroteCodes = boolValue;
}

void StateAuxillary::setMatchPlayerNames()
{
  bool leftTeamNetPlayFound = false;
  bool rightTeamNetPlayFound = false;

  int leftTeamNetPlayPort = -1;
  int rightTeamNetPlayPort = -1;

  std::string leftTeamNetPlayName = "";
  std::string rightTeamNetPlayName = "";

  if (NetPlay::IsNetPlayRunning())
  {
    int currentPortValue = Memory::Read_U16(Metadata::addressControllerPort1);
    if (currentPortValue == 0)
    {
      leftTeamNetPlayPort = 0;
      leftTeamNetPlayFound = true;
    }
    else if (currentPortValue == 1)
    {
      rightTeamNetPlayPort = 0;
      rightTeamNetPlayFound = true;
    }

    currentPortValue = Memory::Read_U16(Metadata::addressControllerPort2);
    if (currentPortValue == 0)
    {
      leftTeamNetPlayPort = 1;
      leftTeamNetPlayFound = true;
    }
    else if (currentPortValue == 1)
    {
      rightTeamNetPlayPort = 1;
      rightTeamNetPlayFound = true;
    }

    // just check this to save read from mem time
    if (!leftTeamNetPlayFound || !rightTeamNetPlayFound)
    {
      currentPortValue = Memory::Read_U16(Metadata::addressControllerPort3);
      if (currentPortValue == 0)
      {
        leftTeamNetPlayPort = 2;
        leftTeamNetPlayFound = true;
      }
      else if (currentPortValue == 1)
      {
        rightTeamNetPlayPort = 2;
        rightTeamNetPlayFound = true;
      }
    }

    // just check this to save read from mem time
    if (!leftTeamNetPlayFound || !rightTeamNetPlayFound)
    {
      currentPortValue = Memory::Read_U16(Metadata::addressControllerPort4);
      if (currentPortValue == 0)
      {
        leftTeamNetPlayPort = 3;
        leftTeamNetPlayFound = true;
      }
      else if (currentPortValue == 1)
      {
        rightTeamNetPlayPort = 3;
        rightTeamNetPlayFound = true;
      }
    }

    std::vector<const NetPlay::Player*> playerArray = Metadata::getPlayerArray();

    if (leftTeamNetPlayFound)
    {
      // netplay is running and we have at least one player on both sides
      // therefore, we can get their player ids from pad map now since we know the ports
      // using the id's we can get their player names

      // getting left team
      // so we're going to go to m_pad_map and find what player id is at port 0
      // then we're going to search for that player id in our player array and return their name
      NetPlay::PlayerId pad_map_player_id = netplayGCMap[leftTeamNetPlayPort];
      // now get the player name that matches the PID we just stored
      for (int j = 0; j < playerArray.size(); j++)
      {
        if (playerArray.at(j)->pid == pad_map_player_id)
        {
          leftTeamNetPlayName = playerArray.at(j)->name;
          break;
        }
      }

      // write the names to their spots in memory
      // need to format name to go .P.o.o.l.B.o.i...
      int index = 0;
      Memory::Write_U8(00, 0x80400018 + index);
      index++;
      for (char& c : leftTeamNetPlayName)
      {
        Memory::Write_U8(c, 0x80400018 + index);
        index++;
        Memory::Write_U8(00, 0x80400018 + index);
        index++;
      }
      for (int i = 0; i < 3; i++)
      {
        Memory::Write_U8(00, 0x80400018 + index);
        index++;
      }
      Memory::Write_U8(01, 0x80400014);
    }
    else
    {
      Memory::Write_U8(00, 0x80400014);
    }

    if (rightTeamNetPlayFound)
    {
      // getting right team
      NetPlay::PlayerId pad_map_player_id = netplayGCMap[rightTeamNetPlayPort];
      for (int j = 0; j < playerArray.size(); j++)
      {
        if (playerArray.at(j)->pid == pad_map_player_id)
        {
          rightTeamNetPlayName = playerArray.at(j)->name;
          break;
        }
      }

      // write the names to their spots in memory
      // need to format name to go .P.o.o.l.B.o.i...
      int index = 0;
      Memory::Write_U8(00, 0x80400040 + index);
      index++;
      for (char& c : rightTeamNetPlayName)
      {
        Memory::Write_U8(c, 0x80400040 + index);
        index++;
        Memory::Write_U8(00, 0x80400040 + index);
        index++;
      }
      for (int i = 0; i < 3; i++)
      {
        Memory::Write_U8(00, 0x80400040 + index);
        index++;
      }
      Memory::Write_U8(01, 0x80400015);
    }
    else
    {
      Memory::Write_U8(00, 0x80400015);
    }
  }
}

void StateAuxillary::saveState(const std::string& filename, bool wait) {
  std::thread t1(&State::SaveAs, filename, wait);
  //State::SaveAs(filename, wait);
  t1.detach();
}

void StateAuxillary::saveStateToTrainingBuffer()
{
  State::SaveToBuffer(training_mode_buffer);
}

void StateAuxillary::saveStateToTrainingBuffer2()
{
  State::SaveMemToBuffer(training_mode_buffer);
}

void StateAuxillary::loadStateFromTrainingBuffer()
{
  State::LoadFromBuffer(training_mode_buffer);
}

void StateAuxillary::loadStateFromTrainingBuffer2()
{
  State::LoadMemFromBuffer(training_mode_buffer);
}

void StateAuxillary::startRecording()
{
  Movie::SetReadOnly(false);
  Movie::ControllerTypeArray controllers{};
  Movie::WiimoteEnabledArray wiimotes{};
  // this is how they're set up in mainwindow.cpp

  if (NetPlay::IsNetPlayRunning())
  {
    for (unsigned int i = 0; i < 4; ++i)
    {
      if (netplayGCMap[i] > 0)
      {
        controllers[i] = Movie::ControllerType::GC;
      }
      else
      {
        controllers[i] = Movie::ControllerType::None;
      }
    }
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      const SerialInterface::SIDevices si_device = Config::Get(Config::GetInfoForSIDevice(i));
      if (si_device == SerialInterface::SIDEVICE_GC_GBA_EMULATED)
        controllers[i] = Movie::ControllerType::GBA;
      else if (SerialInterface::SIDevice_IsGCController(si_device))
        controllers[i] = Movie::ControllerType::GC;
      else
        controllers[i] = Movie::ControllerType::None;
      wiimotes[i] = Config::Get(Config::GetInfoForWiimoteSource(i)) != WiimoteSource::None;
    }
  }
  std::thread t2(&Movie::BeginRecordingInput, controllers, wiimotes);
  t2.detach();
}

void StateAuxillary::stopRecording(const std::string replay_path)
{
  // not moving this to its own thread as of now
  if (Movie::IsRecordingInput())
    Core::RunAsCPUThread([=] {
      Movie::SaveRecording(replay_path);
    });
  if (Movie::IsMovieActive())
  {
    Movie::EndPlayInput(false);
  }
  std::string jsonString = Metadata::getJSONString();
  std::thread t3(&Metadata::writeJSON, jsonString, true);
  t3.detach();
}

// this should be updated to delete the files from where they came from which is not always the citrus replays dir
void StateAuxillary::endPlayback()
{
  /*
  PWSTR path;
  SHGetKnownFolderPath(FOLDERID_Documents, KF_FLAG_DEFAULT, NULL, &path);
  std::wstring strpath(path);
  CoTaskMemFree(path);
  std::string documents_file_path(strpath.begin(), strpath.end());

  std::string replays_path = documents_file_path;
  replays_path += "\\Citrus Replays\\";
  */
  std::string replays_path = File::GetUserPath(D_CITRUSREPLAYS_IDX);
  std::string fileArr[4] = {"output.dtm", "output.dtm.sav", "output.json", "diffFile.patch"};
  for (int i = 0; i < 4; i++)
  {
    std::string innerFileName = replays_path + fileArr[i];
    if (File::Exists(innerFileName))
    {
      std::filesystem::remove(innerFileName);
    }
  }
}

void StateAuxillary::setNetPlayControllers(NetPlay::PadMappingArray m_pad_map, NetPlay::PlayerId m_pid)
{
  netplayGCMap = m_pad_map;
  for (int i = 0; i < 4; i++)
  {
    if (m_pad_map[i] == m_pid)
    {
      ourNetPlayPortsVector.at(i) = 1;
    }
    else
    {
      ourNetPlayPortsVector.at(i) = 0;
    }
  }
}

std::vector<int> StateAuxillary::getOurNetPlayPorts()
{
  return ourNetPlayPortsVector;
}

bool StateAuxillary::isSpectator()
{
  if (!NetPlay::IsNetPlayRunning())
  {
    return false;
  }
  for (int i = 0; i < 4; i++)
  {
    if (ourNetPlayPortsVector.at(i) == 1)
    {
      return false;
    }
  }
  return true;
}

void StateAuxillary::setPrePort(SerialInterface::SIDevices currentPort0,
                                SerialInterface::SIDevices currentPort1,
                                SerialInterface::SIDevices currentPort2,
                                SerialInterface::SIDevices currentPort3)
{
  preMoviePort0 = currentPort0;
  preMoviePort1 = currentPort1;
  preMoviePort2 = currentPort2;
  preMoviePort3 = currentPort3;
}

void StateAuxillary::setPostPort()
{
  Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(0)), preMoviePort0);
  Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(1)), preMoviePort1);
  Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(2)), preMoviePort2);
  Config::SetBaseOrCurrent(Config::GetInfoForSIDevice(static_cast<int>(3)), preMoviePort3);
  SConfig::GetInstance().SaveSettings();
}

bool StateAuxillary::getOverwriteHomeCaptainPositionTrainingMode()
{
  return overwroteHomeCaptainPosTraining;
}

void StateAuxillary::setOverwriteHomeCaptainPositionTrainingMode(bool boolValue)
{
  overwroteHomeCaptainPosTraining = boolValue;
}

bool StateAuxillary::getCustomTrainingModeStart()
{
  return customTrainingModeStart;
}

void StateAuxillary::setCustomTrainingModeStart(bool boolValue)
{
  customTrainingModeStart = boolValue;
}

bool StateAuxillary::getIsRanked()
{
  return isRanked;
}

void StateAuxillary::setIsRanked(bool boolValue)
{
  isRanked = boolValue;
}

int StateAuxillary::getMinPowerPlayPenaltyAmount()
{
  return minPowerPlayPenaltyAmount;
}

void StateAuxillary::hockeyModeInit()
{
  currentTotalScore = 0;
  // reset gecko flags
  Memory::Write_U32(0x00000000, Metadata::addressHockeyModeAttackCharacterId);
  Memory::Write_U8(0, Metadata::addressHockeyModePenaltyFlag);
  // reset and init team vectors
  hockeyLeftTeamCharacterInfo.clear();
  /* good ones for horizontal tunnel. would be better if we stacked them vertical tho
  std::vector<float> leftPenaltyXAddresses = {-1, -.5, .5, 1};
  std::vector<float> rightPenaltyXAddresses = {15, 16, 17, 18};
  std::vector<float> penaltyYAddresses = {15, 15, 15, 15};
  */
  // in their goals
  /*
  std::vector<float> leftPenaltyXAddresses = {-23, -23, -23, -23};
  std::vector<float> rightPenaltyXAddresses = {23, 23, 23, 23};
  std::vector<float> penaltyYAddresses = {-.75, -.25, .25, .75};
  */
  // vertical tunnel stack
  
  std::vector<float> leftPenaltyXAddresses = {-1, -1, -1, -1};
  std::vector<float> rightPenaltyXAddresses = {.5, .5, .5, .5};
  std::vector<float> penaltyYAddresses = {15, 17, 19, 20};
  
  // bench stack
  /*
  std::vector<float> leftPenaltyXAddresses = {-12, -11, -10, -9};
  std::vector<float> rightPenaltyXAddresses = {12, 11, 10, 9};
  std::vector<float> penaltyYAddresses = {14, 14, 14, 14};
  */
  for (int i = 0; i < 4; i++)
  {
    u32 characterPointer = Memory::Read_U32(Metadata::addressCharacterPointersBase + (i * 4));
    HockeyCharacterInfo characterInfo = {false, 20, leftPenaltyXAddresses.at(i), penaltyYAddresses.at(i)};
    hockeyLeftTeamCharacterInfo.emplace(characterPointer, characterInfo);
  }
  hockeyRightTeamCharacterInfo.clear();
  for (int i = 0; i < 4; i++)
  {
    u32 characterPointer = Memory::Read_U32(Metadata::addressCharacterPointersBase + 16 + (i * 4));
    HockeyCharacterInfo characterInfo = {false, 20, rightPenaltyXAddresses.at(i), penaltyYAddresses.at(i)};
    hockeyRightTeamCharacterInfo.emplace(characterPointer, characterInfo);
  }
}

std::map<u32, StateAuxillary::HockeyCharacterInfo> StateAuxillary::getHockeyLeftTeamCharacterInfo()
{
  return hockeyLeftTeamCharacterInfo;
}

std::map<u32, StateAuxillary::HockeyCharacterInfo> StateAuxillary::getHockeyRightTeamCharacterInfo()
{
  return hockeyRightTeamCharacterInfo;
}

void StateAuxillary::setHockeyLeftTeamCharacterInfo(u32 characterPointer, HockeyCharacterInfo characterInfo)
{
  hockeyLeftTeamCharacterInfo[characterPointer] = characterInfo;
}

void StateAuxillary::setHockeyRightTeamCharacterInfo(u32 characterPointer, HockeyCharacterInfo characterInfo)
{
  hockeyRightTeamCharacterInfo[characterPointer] = characterInfo;
}

int StateAuxillary::getHockeyLeftTeamTotalPenalties()
{
  return hockeyLeftTeamTotalPenalties;
}

int StateAuxillary::getHockeyRightTeamTotalPenalties()
{
  return hockeyRightTeamTotalPenalties;
}

void StateAuxillary::setHockeyLeftTeamTotalPenalties(int hitParam)
{
  hockeyLeftTeamTotalPenalties = hitParam;
}

void StateAuxillary::setHockeyRightTeamTotalPenalties(int hitParam)
{
  hockeyRightTeamTotalPenalties = hitParam;
}

void StateAuxillary::updateHockeyDisplay()
{
  u16 totalScore = Memory::Read_U16(Metadata::addressLeftSideScore) +
                   Memory::Read_U16(Metadata::addressRightSideScore);
  if (totalScore != currentTotalScore)
  {
    // reset all positions
    hockeyModeInit();
    currentTotalScore = totalScore;
    return;
  }
  std::string leftTeamDisplayString =
      fmt::format("Left Team Total Penalties: {}\n", hockeyLeftTeamTotalPenalties);
  std::string rightTeamDisplayString =
      fmt::format("Right Team Total Penalties: {}\n", hockeyRightTeamTotalPenalties);
  /*
  OSD::AddTypedMessage(OSD::MessageType::HockeyLeftTeamTotalPenalties,
                       fmt::format("Left Team Total Penalties: {}\nTest new line", hockeyLeftTeamTotalPenalties),
                       OSD::Duration::SHORT, OSD::Color::CYAN);
  OSD::AddTypedMessage(OSD::MessageType::HockeyRightTeamTotalPenalties,
                       fmt::format("Right Team Total Penalties: {}", hockeyRightTeamTotalPenalties),
                       OSD::Duration::SHORT, OSD::Color::YELLOW);
  */
  int i = 0;
  int j = 1;
  if (hockeyLeftTeamTotalPenalties >= 1)
  {
    for (auto& p : StateAuxillary::getHockeyLeftTeamCharacterInfo())
    {
      // we should add a condition that the game is not paused
      if (p.second.currentlyPenalized)
      {
        if (p.second.currentlyPenalizedTimeRemaining > 0)
        {
          PowerPC::HostWrite_F32(p.second.penaltyXAddress, p.first + 0x520);
          PowerPC::HostWrite_F32(p.second.penaltyYAddress, p.first + 0x524);
          INFO_LOG_FMT(CORE, "Player {} should have been sentenced to {}, {}", p.first,
                       p.second.penaltyXAddress, p.second.penaltyYAddress);
          INFO_LOG_FMT(CORE, "Player {} was sentenced to {}, {}", p.first,
                       Memory::Read_U32(p.first + 0x520), Memory::Read_U32(p.first + 0x524));
          float remainingTime = p.second.currentlyPenalizedTimeRemaining - (float(1) / float(60));
          setHockeyLeftTeamCharacterInfo(p.first, {p.second.currentlyPenalized, remainingTime, p.second.penaltyXAddress, p.second.penaltyYAddress});
          std::string playerDisplayString = fmt::format(
              "\nPlayer {} Penalty Time Remaining: {} seconds", j + 1, std::round(remainingTime));
          leftTeamDisplayString += playerDisplayString;
          j++;
          /*
          OSD::AddTypedMessage(OSD::MessageType(3 + i),
                               fmt::format("Left Team Player {} Penalty Time Remaining: {} seconds",
                                           i + 1, std::round(remainingTime)),
                               OSD::Duration::SHORT, OSD::Color::CYAN);
          */
        }
        else
        {
          setHockeyLeftTeamCharacterInfo(p.first, {false, 20, p.second.penaltyXAddress, p.second.penaltyYAddress});
          PowerPC::HostWrite_F32(0, p.first + 0x520);
          PowerPC::HostWrite_F32(0, p.first + 0x524);
        }
      }
      i++;
    }
    
  }
  i = 0;
  j = 1;
  if (hockeyRightTeamTotalPenalties >= 1)
  {
    for (auto& p : StateAuxillary::getHockeyRightTeamCharacterInfo())
    {
      if (p.second.currentlyPenalized)
      {
        if (p.second.currentlyPenalizedTimeRemaining > 0)
        {
          PowerPC::HostWrite_F32(p.second.penaltyXAddress, p.first + 0x520);
          PowerPC::HostWrite_F32(p.second.penaltyYAddress, p.first + 0x524);
          INFO_LOG_FMT(CORE, "Player {} should have been sentenced to {}, {}", p.first,
                       p.second.penaltyXAddress, p.second.penaltyYAddress);
          INFO_LOG_FMT(CORE, "Player {} was sentenced to {}, {}", p.first,
                       Memory::Read_U32(p.first + 0x520), Memory::Read_U32(p.first + 0x524));
          float remainingTime = p.second.currentlyPenalizedTimeRemaining - (float(1) / float(60));
          setHockeyRightTeamCharacterInfo(p.first,
                                          {p.second.currentlyPenalized, remainingTime,
                                           p.second.penaltyXAddress, p.second.penaltyYAddress});
          std::string playerDisplayString = fmt::format(
              "\nPlayer {} Penalty Time Remaining: {} seconds", j + 1, std::round(remainingTime));
          rightTeamDisplayString += playerDisplayString;
          j++;
          /*
          OSD::AddTypedMessage(OSD::MessageType(6 + i),
                               fmt::format("Right Team Player {} Penalty Time Remaining: {} seconds", i + 1,
                                std::round(remainingTime)),
                               OSD::Duration::SHORT, OSD::Color::YELLOW);
          */
        }
        else
        {
          setHockeyRightTeamCharacterInfo(p.first, {false, 20, p.second.penaltyXAddress, p.second.penaltyYAddress});
          PowerPC::HostWrite_F32(0, p.first + 0x520);
          PowerPC::HostWrite_F32(0, p.first + 0x524);
        }
      }
      i++;
    }
  }
  OSD::AddTypedMessage(OSD::MessageType::HockeyLeftTeamTotalPenalties, leftTeamDisplayString,
                       OSD::Duration::SHORT, OSD::Color::CYAN);
  OSD::AddTypedMessage(OSD::MessageType::HockeyRightTeamTotalPenalties, rightTeamDisplayString,
                       OSD::Duration::SHORT, OSD::Color::YELLOW);
}

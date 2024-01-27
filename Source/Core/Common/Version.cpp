// Copyright 2008 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Common/Version.h"

#include <string>

#include "Common/scmrev.h"

namespace Common
{
#ifdef _DEBUG
#define BUILD_TYPE_STR "Debug "
#elif defined DEBUGFAST
#define BUILD_TYPE_STR "DebugFast "
#else
#define BUILD_TYPE_STR ""
#endif
#define CITRUS_REV_STR "0.2.0.4"

const std::string& GetScmRevStr()
{
  static const std::string scm_rev_str = "Citrus Dolphin "
#if !SCM_IS_MASTER
                                         "[" SCM_BRANCH_STR "] "
#endif

#ifdef __INTEL_COMPILER
      BUILD_TYPE_STR CITRUS_REV_STR "-ICC";
#else
      BUILD_TYPE_STR CITRUS_REV_STR;
#endif
  return scm_rev_str;
}

const std::string& GetScmRevGitStr()
{
  static const std::string scm_rev_git_str = SCM_REV_STR;
  return scm_rev_git_str;
}

const std::string& GetCitrusRevStr()
{
  static const std::string rio_rev_str = CITRUS_REV_STR;
  return rio_rev_str;
}

const std::string& GetScmDescStr()
{
  static const std::string scm_desc_str = CITRUS_REV_STR;
  return scm_desc_str;
}

const std::string& GetScmBranchStr()
{
  static const std::string scm_branch_str = SCM_BRANCH_STR;
  return scm_branch_str;
}

const std::string& GetScmDistributorStr()
{
  static const std::string scm_distributor_str = SCM_DISTRIBUTOR_STR;
  return scm_distributor_str;
}

const std::string& GetScmUpdateTrackStr()
{
  static const std::string scm_update_track_str = SCM_UPDATE_TRACK_STR;
  return scm_update_track_str;
}

const std::string& GetNetplayDolphinVer()
{
#ifdef _WIN32
  static const std::string netplay_dolphin_ver = "Citrus " CITRUS_REV_STR " Win";
#elif __APPLE__
  static const std::string netplay_dolphin_ver = "Citrus " CITRUS_REV_STR " Mac";
#else
  static const std::string netplay_dolphin_ver = "Citrus " CITRUS_REV_STR " Lin";
#endif
  return netplay_dolphin_ver;
}

}  // namespace Common

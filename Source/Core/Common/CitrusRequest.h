#pragma once
#include <string>
#include <cstdint>
#include <map>

class CitrusRequest
{
public:
  // the general idea here is that we have a repeatable way of making calls to the citrus server
  // and the only thing changing is the endpoint and contents

  enum LoginError : uint8_t
  {
    NoError = 0,
    NoUserFile = 1,
    InvalidUserId = 2,
    InvalidLogin = 3,
    ServerError = 4
  };

  static std::map<CitrusRequest::LoginError, std::string> loginErrorMap;

  static std::string GetCurrentBaseURL();
  static LoginError LogInUser(std::string userId, std::string jwt);
  static void SendMatchStats(std::string matchJSON);
};

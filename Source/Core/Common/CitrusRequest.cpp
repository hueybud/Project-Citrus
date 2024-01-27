#include "CitrusRequest.h"
#include "HttpRequest.h"
#include <picojson.h>
#include "Logging/Log.h"
#include <Core/Metadata.h>

static bool PROD_ENABLED = true;

static Common::HttpRequest::Headers headers = {
    {"user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like "
                   "Gecko) Chrome/97.0.4692.71 Safari/537.36"},
    {"content-type", "application/json"}};

std::map<CitrusRequest::LoginError, std::string> CitrusRequest::loginErrorMap = {
    {CitrusRequest::LoginError::NoUserFile, "No user file found"},
    {CitrusRequest::LoginError::InvalidUserId, "The User ID in your user.json file is not formatted correctly. \n\nConsider logging out and logging back in via Citrus Launcher"},
    {CitrusRequest::LoginError::InvalidLogin, "Your user.json file exists but failed to authenticate you. \n\nConsider logging out and logging back in via Citrus Launcher"},
    {CitrusRequest::LoginError::ServerError, "We encounterd an error with the server while trying to log you in."},
};

std::string CitrusRequest::GetCurrentBaseURL()
{
  if (PROD_ENABLED)
  {
    return "http://23.22.39.40:8080";
  }

  return "https://localhost:3000";
  
}

CitrusRequest::LoginError CitrusRequest::LogInUser(std::string userId, std::string jwt)
{
  // The timeout is only for trying to establish the connection to the server
  // If we don't establish a connection with 3 seconds, we error.
  // If we do establish a connection, we are synchronously waiting for the response.
  // We can avoid waiting for a response for other requests by simply not storing the result to a response variable

  Common::HttpRequest req{std::chrono::seconds{3}};

  picojson::object jsonRequestObj;
  jsonRequestObj["userId"] = picojson::value(userId);
  jsonRequestObj["jwt"] = picojson::value(jwt);

  std::string jsonRequestString = picojson::value(jsonRequestObj).serialize();

  std::string loginURL = GetCurrentBaseURL() + "/citrus/" + "verifyLogIn";
  auto resp = req.Post(loginURL, jsonRequestString, headers, Common::HttpRequest::AllowedReturnCodes::All, true);
  if (!resp)
  {
    INFO_LOG_FMT(COMMON, "Login request could not communicate with server");
    return LoginError::ServerError;
  }

  const std::string contents(reinterpret_cast<char*>(resp->data()), resp->size());
  INFO_LOG_FMT(COMMON, "Login request JSON response: {}", contents);

  picojson::value json;
  const std::string err = picojson::parse(json, contents); 
  if (!err.empty())
  {
    INFO_LOG_FMT(COMMON, "Invalid JSON received from login request: {}", err);
    return LoginError::ServerError;
  }
  picojson::object obj = json.get<picojson::object>();

  // Technically someone can intercept this response with Fiddler or a similar app and change the "valid" value to true
  // even if they use the wrong log in
  // However, we authenticate server side when they send a request to submit match info anyways
  // and will not take the data if the jwt does not validate with the user id and private-key signing
  // So this is really just a preliminary auth check for 99.99% of users (looking at you, Wes)
  if (obj["valid"].get<bool>() == true)
  {
    INFO_LOG_FMT(COMMON, "Login successful");
    return LoginError::NoError;
  }

  if (obj["error"].get<std::string>() == "Unable to decode JWT against the private key")
  {
    return LoginError::InvalidLogin;
  }

  return LoginError::ServerError;
}

void CitrusRequest::SendMatchStats(std::string matchJSON)
{
  Common::HttpRequest req{std::chrono::seconds{3}};

  picojson::object jsonRequestObj;

  jsonRequestObj["userId"] = picojson::value(Metadata::getCitrusUser().GetUserInfo().userId);
  jsonRequestObj["jwt"] = picojson::value(Metadata::getCitrusUser().GetUserInfo().jwt);
  jsonRequestObj["matchData"] = picojson::value(matchJSON);
  std::string playerIdString = "";

  //jsonRequestObj["players"] = picojson::value(&Metadata::getPlayerArray()[0]);

  std::string jsonRequestString = picojson::value(jsonRequestObj).serialize();

  std::string requestURL = GetCurrentBaseURL() + "/citrus/" + "submitMatchData";
  req.Post(requestURL, jsonRequestString, headers, Common::HttpRequest::AllowedReturnCodes::All,
           true);
  INFO_LOG_FMT(COMMON, "Submitted match data");
}

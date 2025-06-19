#pragma once

#include <config.h>

#include <Common/StringUtils.h>
#include <Poco/Net/Context.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/Timestamp.h>
#include <Poco/URI.h>

#include <iosfwd>
#include <map>
#include <memory>
#include <string>

namespace DB
{

struct AuthEndpoints
{
    std::string auth_url;
    std::string client_id;
    std::string api_host;
};

static const std::map<std::string, AuthEndpoints> managed_service_endpoints = {
    {
        ".clickhouse-dev.com",
        {
            "https://ch-local-dev.us.auth0.com",
            "HxCYHJpfUCQtX6Y8EzFWImHLVFbVNahh",
            "https://console-api.clickhouse-dev.com"
        }
    }
};

inline const AuthEndpoints * getAuthEndpoints(const std::string & host)
{
    for (const auto & [suffix, endpoints] : managed_service_endpoints)
    {
        if (endsWith(host, suffix))
            return &endpoints;
    }
    return nullptr;
}

class JwtProvider
{
public:
    JwtProvider(
        std::string auth_url,
        std::string client_id,
        std::ostream & out,
        std::ostream & err);
    virtual ~JwtProvider() = default;

    /// Returns a valid ClickHouse JWT.
    /// Implementations are responsible for handling the entire lifecycle,
    /// including initial login and subsequent refreshes.
    /// @return The ClickHouse JWT on success, or an empty string on failure.
    virtual std::string getJWT() = 0;

protected:
    bool initialLogin();
    bool refreshIdPAccessToken();

    // Configuration
    std::string auth_url_str;
    std::string client_id_str;
    std::ostream & output_stream;
    std::ostream & error_stream;

    // Token State
    std::string idp_access_token;
    std::string idp_refresh_token;
    Poco::Timestamp idp_access_token_expires_at{0};

protected:
    static std::unique_ptr<Poco::Net::HTTPClientSession> createHTTPSession(const Poco::URI & uri);
    static bool openURLInBrowser(const std::string & url);
    static Poco::Timestamp getJwtExpiry(const std::string & token);
};

/// Creates the appropriate JWT provider based on the application configuration.
std::unique_ptr<JwtProvider> createJwtProvider(
    const std::string & auth_url,
    const std::string & client_id,
    const std::string & host,
    std::ostream & out,
    std::ostream & err);

}

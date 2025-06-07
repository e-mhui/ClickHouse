#pragma once

#include <config.h>

#include <Poco/Net/Context.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/Timestamp.h>
#include <Poco/URI.h>

#include <iosfwd>
#include <memory>
#include <string>

namespace DB
{

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
    const std::string & control_plane_url,
    std::ostream & out,
    std::ostream & err);

}

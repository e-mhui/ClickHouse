#include <Client/CloudJwtProvider.h>
#include <Common/Exception.h>
#include <Common/StringUtils.h>
#include <config.h>

#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPSClientSession.h>
#include <Poco/Net/Context.h>
#include <Poco/Net/SSLManager.h>
#include <Poco/StreamCopier.h>
#include <Poco/URI.h>
#include <Poco/JSON/Parser.h>
#include <Poco/JSON/Object.h>
#include <Poco/Dynamic/Var.h>

#include <thread>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <jwt-cpp/jwt.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int BAD_ARGUMENTS;
    extern const int SUPPORT_IS_DISABLED;
    extern const int NETWORK_ERROR;
}

namespace
{
}

CloudJwtProvider::CloudJwtProvider(
    std::string auth_url, std::string client_id, std::string host,
    std::ostream & out, std::ostream & err)
    : JwtProvider(std::move(auth_url), std::move(client_id), out, err),
      host_str(std::move(host)) {}

std::string CloudJwtProvider::getJWT()
{
    Poco::Timestamp now;
    Poco::Timestamp expiration_buffer = 5 * Poco::Timespan::SECONDS;

    if (!final_clickhouse_jwt.empty() && now < final_clickhouse_jwt_expires_at - expiration_buffer)
        return final_clickhouse_jwt;

    if (!idp_refresh_token.empty())
    {
        if (refreshIdPAccessToken() && swapIdPTokenForClickHouseJWT())
            return final_clickhouse_jwt;
    }

    if (initialLogin() && swapIdPTokenForClickHouseJWT())
        return final_clickhouse_jwt;

    error_stream << "Failed to obtain a valid ClickHouse JWT." << std::endl;
    return "";
}

bool CloudJwtProvider::swapIdPTokenForClickHouseJWT()
{
    const auto * endpoints = getAuthEndpoints(host_str);

    if (!endpoints)
    {
        error_stream << "Error: cannot determine token swap endpoint from hostname " << host_str
                     << ". Please use a managed ClickHouse hostname." << std::endl;
        return false;
    }

    std::string swap_url = endpoints->api_host + "/token-swap";

    output_stream << "Swapping IdP token for a ClickHouse JWT..." << std::endl;
    try
    {
        Poco::URI swap_uri(swap_url);
        swap_uri.addQueryParameter("hostname", host_str);
        auto session = createHTTPSession(swap_uri);
        Poco::Net::HTTPRequest request(Poco::Net::HTTPRequest::HTTP_POST, swap_uri.getPathAndQuery(), Poco::Net::HTTPMessage::HTTP_1_1);
        request.set("Authorization", "Bearer " + idp_access_token);
        request.setContentLength(0);
        session->sendRequest(request);

        Poco::Net::HTTPResponse response;
        std::istream & rs = session->receiveResponse(response);
        if (response.getStatus() != Poco::Net::HTTPResponse::HTTP_OK)
        {
            std::string error_body;
            Poco::StreamCopier::copyToString(rs, error_body);
            error_stream << "Error swapping token: " << response.getStatus() << " " << response.getReason() << "\nResponse: " << error_body << std::endl;
            return false;
        }

        Poco::JSON::Object::Ptr object = Poco::JSON::Parser().parse(rs).extract<Poco::JSON::Object::Ptr>();
        final_clickhouse_jwt = object->getValue<std::string>("token");
        final_clickhouse_jwt_expires_at = getJwtExpiry(final_clickhouse_jwt);

        output_stream << "Successfully obtained ClickHouse JWT." << std::endl;
        return true;
    }
    catch(const Poco::Exception & ex)
    {
        error_stream << "Exception during token swap: " << ex.displayText() << std::endl;
        return false;
    }
}

}

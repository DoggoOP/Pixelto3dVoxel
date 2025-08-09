#include "geo_utils.h"
#include <cmath>

namespace {
// WGS84 ellipsoid constants
constexpr double a = 6378137.0;               // semi-major axis
constexpr double f = 1.0 / 298.257223563;     // flattening
constexpr double b = a * (1 - f);             // semi-minor axis
constexpr double e2 = f * (2 - f);            // first eccentricity squared
constexpr double eps = e2 / (1.0 - e2);       // second eccentricity squared

cv::Vec3d geodeticToEcef(double lat, double lon, double alt) {
    double sinLat = std::sin(lat);
    double cosLat = std::cos(lat);
    double sinLon = std::sin(lon);
    double cosLon = std::cos(lon);
    double N = a / std::sqrt(1.0 - e2 * sinLat * sinLat);
    double x = (N + alt) * cosLat * cosLon;
    double y = (N + alt) * cosLat * sinLon;
    double z = (N * (1.0 - e2) + alt) * sinLat;
    return {x, y, z};
}

cv::Vec3d ecefToGeodetic(const cv::Vec3d& ecef) {
    double x = ecef[0];
    double y = ecef[1];
    double z = ecef[2];
    double p = std::sqrt(x*x + y*y);
    double th = std::atan2(a * z, b * p);
    double lon = std::atan2(y, x);
    double lat = std::atan2(z + eps * b * std::pow(std::sin(th),3),
                            p - e2 * a * std::pow(std::cos(th),3));
    double N = a / std::sqrt(1.0 - e2 * std::sin(lat) * std::sin(lat));
    double alt = p / std::cos(lat) - N;
    return {lat * 180.0 / CV_PI, lon * 180.0 / CV_PI, alt};
}
}

GeoRef makeGeoRef(double lat0_deg, double lon0_deg, double alt0_m) {
    GeoRef ref;
    ref.lat0_deg = lat0_deg;
    ref.lon0_deg = lon0_deg;
    ref.alt0_m   = alt0_m;
    double lat0 = lat0_deg * CV_PI / 180.0;
    double lon0 = lon0_deg * CV_PI / 180.0;
    ref.ecef0 = geodeticToEcef(lat0, lon0, alt0_m);
    double sinLat = std::sin(lat0);
    double cosLat = std::cos(lat0);
    double sinLon = std::sin(lon0);
    double cosLon = std::cos(lon0);
    ref.ecef_to_enu = {
        -sinLon,           cosLon,            0,
        -sinLat*cosLon, -sinLat*sinLon, cosLat,
        cosLat*cosLon,  cosLat*sinLon, sinLat
    };
    ref.enu_to_ecef = ref.ecef_to_enu.t();
    return ref;
}

cv::Vec3d geodeticToEnu(const GeoRef& ref, double lat_deg, double lon_deg, double alt_m) {
    double lat = lat_deg * CV_PI / 180.0;
    double lon = lon_deg * CV_PI / 180.0;
    cv::Vec3d ecef = geodeticToEcef(lat, lon, alt_m);
    cv::Vec3d diff = ecef - ref.ecef0;
    return ref.ecef_to_enu * diff;
}

cv::Vec3d enuToGeodetic(const GeoRef& ref, const cv::Vec3d& enu) {
    cv::Vec3d ecef = ref.enu_to_ecef * enu + ref.ecef0;
    return ecefToGeodetic(ecef);
}

#pragma once
#include <opencv2/core.hpp>

// Geodetic <-> local ENU conversions using WGS84 ellipsoid
struct GeoRef {
    double lat0_deg; // reference latitude  degrees
    double lon0_deg; // reference longitude degrees
    double alt0_m;   // reference altitude  metres
    cv::Vec3d ecef0; // reference in ECEF metres
    cv::Matx33d ecef_to_enu;
    cv::Matx33d enu_to_ecef;
};

GeoRef makeGeoRef(double lat0_deg, double lon0_deg, double alt0_m);
cv::Vec3d geodeticToEnu(const GeoRef& ref, double lat_deg, double lon_deg, double alt_m);
cv::Vec3d enuToGeodetic(const GeoRef& ref, const cv::Vec3d& enu);

#pragma once
#include <opencv2/core.hpp>
#include <cmath>

inline cv::Matx33f yprDegreesToMatrix(const cv::Vec3f& ypr) {
    const float deg2rad = static_cast<float>(CV_PI / 180.0);
    float yaw   = (90.0f - ypr[0]) * deg2rad; // convert to ENU convention
    float pitch =  ypr[1] * deg2rad;
    float roll  =  ypr[2] * deg2rad;

    float cy = cosf(yaw),   sy = sinf(yaw);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cr = cosf(roll),  sr = sinf(roll);
    cv::Matx33f Rz = {cy, -sy, 0, sy, cy, 0, 0, 0, 1};
    cv::Matx33f Ry = {cp, 0, sp, 0, 1, 0, -sp, 0, cp};
    cv::Matx33f Rx = {1, 0, 0, 0, cr, -sr, 0, sr, cr};
    return Rz * Ry * Rx; // yaw→pitch→roll (Z‑Y‑X)
}

// Convert WGS‑84 latitude/longitude/altitude to local ENU metres
// using a simple equirectangular approximation. The reference
// latitude/longitude/altitude defines the origin of the local frame.
inline cv::Vec3f geodeticToENU(float lat_deg, float lon_deg, float alt_m,
                              float ref_lat_deg, float ref_lon_deg, float ref_alt_m) {
    const float R = 6378137.0f; // mean Earth radius (m)
    const float deg2rad = static_cast<float>(CV_PI / 180.0);
    float dlat = (lat_deg - ref_lat_deg) * deg2rad;
    float dlon = (lon_deg - ref_lon_deg) * deg2rad;
    float east  = R * dlon * cosf(ref_lat_deg * deg2rad);
    float north = R * dlat;
    float up    = alt_m - ref_alt_m;
    return {east, north, up};
}
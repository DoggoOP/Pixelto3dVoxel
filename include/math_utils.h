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
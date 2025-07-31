#pragma once
#include <opencv2/core.hpp>
#include <cmath>

inline cv::Matx33f yprDegreesToMatrix(const cv::Vec3f& ypr) {
    const float deg2rad = static_cast<float>(CV_PI / 180.0);
    float cy = cosf(ypr[0] * deg2rad), sy = sinf(ypr[0] * deg2rad);
    float cp = cosf(ypr[1] * deg2rad), sp = sinf(ypr[1] * deg2rad);
    float cr = cosf(ypr[2] * deg2rad), sr = sinf(ypr[2] * deg2rad);
    cv::Matx33f Rz = {cy, -sy, 0, sy, cy, 0, 0, 0, 1};
    cv::Matx33f Ry = {cp, 0, sp, 0, 1, 0, -sp, 0, cp};
    cv::Matx33f Rx = {1, 0, 0, 0, cr, -sr, 0, sr, cr};
    return Rz * Ry * Rx; // yaw→pitch→roll (Z‑Y‑X)
}
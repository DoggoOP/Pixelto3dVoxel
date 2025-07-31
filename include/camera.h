#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct Camera {
    std::string id;
    std::string folder;
    cv::Vec3f   position;          // metres
    cv::Vec3f   ypr_deg;           // yaw, pitch, roll in degrees
    float       fov_deg;           // horizontal FOV
    std::vector<std::string> frames; // absolute image paths

    // 3×3 body‑to‑world rotation (East‑North‑Up)
    cv::Matx33f rotation() const;
};
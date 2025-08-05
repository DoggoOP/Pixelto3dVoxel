#pragma once
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>

struct VoxelGrid {
    int         N;             // side length
    float       voxelSize;     // metres
    cv::Vec3f   center;        // world coords of grid centre
    std::vector<float> data;   // N^3 values

    VoxelGrid(int n, float v, cv::Vec3f c)
        : N(n), voxelSize(v), center(c), data(n*n*n, 0.0f) {}

    inline int idx(int ix, int iy, int iz) const {
        return (iz * N + iy) * N + ix;
    }
    inline void add(int ix, int iy, int iz, float v) {
        data[idx(ix, iy, iz)] += v;
    }
    inline void clear() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
};
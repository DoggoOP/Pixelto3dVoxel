#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include "camera.h"
#include "math_utils.h"
#include "voxel_grid.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// --- forward declarations --------------------------------------------------
static void load_metadata(const std::string& path, std::vector<Camera>& cams, VoxelGrid& grid);
static void process_camera(const Camera& cam, VoxelGrid& grid, float motionThreshold);
static void cast_ray(const cv::Vec3f& camPos, const cv::Vec3f& dir, VoxelGrid& grid, float pixVal);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: pixeltovoxel <metadata.json>" << std::endl;
        return 1;
    }
    std::vector<Camera> cams;
    VoxelGrid grid(1,1.0f,{0,0,0}); // temp, will be replaced in load
    load_metadata(argv[1], cams, grid);

    const float MOTION_THRESHOLD = 2.0f;
    for (const auto& cam : cams) {
        process_camera(cam, grid, MOTION_THRESHOLD);
    }

    // write raw grid to binary
    std::ofstream out("rays.bin", std::ios::binary);
    out.write(reinterpret_cast<char*>(&grid.N), sizeof(int));
    out.write(reinterpret_cast<char*>(&grid.voxelSize), sizeof(float));
    out.write(reinterpret_cast<char*>(&grid.center), sizeof(float)*3);
    out.write(reinterpret_cast<char*>(grid.data.data()), sizeof(float)*grid.data.size());
    out.close();
    return 0;
}

// ---------------------------------------------------------------------------
static void load_metadata(const std::string& path, std::vector<Camera>& cams, VoxelGrid& grid) {
    std::ifstream f(path);
    json j; f >> j;
    int N = j["voxel"]["N"].get<int>();
    float vs = j["voxel"]["voxel_size"].get<float>();
    auto c = j["voxel"]["center"];
    grid = VoxelGrid(N, vs, {c[0].get<float>(), c[1].get<float>(), c[2].get<float>()});

    for (auto& jc : j["cameras"]) {
        Camera cam;
        cam.id        = jc["id"].get<std::string>();
        cam.folder    = jc["folder"].get<std::string>();
        cam.position  = { jc["position"][0].get<float>(), jc["position"][1].get<float>(), jc["position"][2].get<float>() };
        cam.ypr_deg   = { jc["yaw_pitch_roll"][0].get<float>(), jc["yaw_pitch_roll"][1].get<float>(), jc["yaw_pitch_roll"][2].get<float>() };
        cam.fov_deg   = jc["fov_degrees"].get<float>();

        // collect frames (sorted)
        std::vector<std::string> frames;
        for (auto& p : fs::directory_iterator(cam.folder)) {
            if (p.path().extension() == ".jpg" || p.path().extension() == ".png")
                frames.push_back(p.path().string());
        }
        std::sort(frames.begin(), frames.end());
        cam.frames = std::move(frames);
        cams.push_back(cam);
    }
}

// ---------------------------------------------------------------------------
static void process_camera(const Camera& cam, VoxelGrid& grid, float motionThreshold) {
    if (cam.frames.size() < 2) return;
    cv::Mat prev;
    const float focal_px = 0.5f * cam.fov_deg * CV_PI / 180.0f; // approximate

    const cv::Matx33f Rcw = cam.rotation();

    for (size_t i = 0; i < cam.frames.size(); ++i) {
        cv::Mat img = cv::imread(cam.frames[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;
        if (prev.empty()) { prev = img; continue; }
        cv::Mat diff;
        cv::absdiff(img, prev, diff);
        prev = img;
        for (int v = 0; v < diff.rows; ++v) {
            const uchar* row = diff.ptr<uchar>(v);
            for (int u = 0; u < diff.cols; ++u) {
                if (row[u] < motionThreshold) continue;
                // pixel to camera‑local ray (pinhole)
                float x = (u - 0.5f*diff.cols);
                float y = -(v - 0.5f*diff.rows);
                float z = -focal_px;
                cv::Vec3f dir_cam = cv::normalize(cv::Vec3f(x, y, z));
                cv::Vec3f dir_world = Rcw * dir_cam;
                cast_ray(cam.position, dir_world, grid, static_cast<float>(row[u]));
            }
        }
    }
}

// 3‑D DDA traversal ----------------------------------------------------------
static void cast_ray(const cv::Vec3f& camPos, const cv::Vec3f& dir, VoxelGrid& grid, float pixVal) {
    // compute entry point into bounding box
    const float half = 0.5f * grid.N * grid.voxelSize;
    cv::Vec3f boxMin = grid.center - cv::Vec3f(half, half, half);
    cv::Vec3f boxMax = grid.center + cv::Vec3f(half, half, half);

    cv::Vec3f tMin, tMax;
    for (int i = 0; i < 3; ++i) {
        float invD = 1.0f / dir[i];
        tMin[i] = (boxMin[i] - camPos[i]) * invD;
        tMax[i] = (boxMax[i] - camPos[i]) * invD;
        if (tMin[i] > tMax[i]) std::swap(tMin[i], tMax[i]);
    }
    float tEnter = std::max({tMin[0], tMin[1], tMin[2]});
    float tExit  = std::min({tMax[0], tMax[1], tMax[2]});
    if (tEnter > tExit) return; // no hit

    cv::Vec3f p = camPos + dir * std::max(tEnter, 0.0f);

    int ix = static_cast<int>((p[0]-boxMin[0]) / grid.voxelSize);
    int iy = static_cast<int>((p[1]-boxMin[1]) / grid.voxelSize);
    int iz = static_cast<int>((p[2]-boxMin[2]) / grid.voxelSize);
    cv::Vec3f step = {
        dir[0] > 0 ? 1.0f : -1.0f,
        dir[1] > 0 ? 1.0f : -1.0f,
        dir[2] > 0 ? 1.0f : -1.0f
    };
    cv::Vec3f nextVoxelBoundary = {
        boxMin[0] + (ix + (step[0]>0)) * grid.voxelSize,
        boxMin[1] + (iy + (step[1]>0)) * grid.voxelSize,
        boxMin[2] + (iz + (step[2]>0)) * grid.voxelSize };

    cv::Vec3f invDir{1.0f / dir[0], 1.0f / dir[1], 1.0f / dir[2]};
    cv::Vec3f tMaxV  = (nextVoxelBoundary - camPos).mul(invDir);
    cv::Vec3f tDelta = cv::abs(cv::Vec3f(grid.voxelSize, grid.voxelSize, grid.voxelSize).mul(invDir));


    const int MAX_STEPS = grid.N*3; // conservative
    for (int stepCount=0; stepCount<MAX_STEPS; ++stepCount) {
        if (ix<0||iy<0||iz<0||ix>=grid.N||iy>=grid.N||iz>=grid.N) break;
        grid.add(ix,iy,iz,pixVal);
        // advance to next voxel
        if (tMaxV[0] < tMaxV[1]) {
            if (tMaxV[0] < tMaxV[2]) { ix += step[0]; tMaxV[0] += tDelta[0]; }
            else                      { iz += step[2]; tMaxV[2] += tDelta[2]; }
        } else {
            if (tMaxV[1] < tMaxV[2]) { iy += step[1]; tMaxV[1] += tDelta[1]; }
            else                      { iz += step[2]; tMaxV[2] += tDelta[2]; }
        }
    }
}

// --- Camera rotation implementation ---------------------------------------
cv::Matx33f Camera::rotation() const { return yprDegreesToMatrix(ypr_deg); }
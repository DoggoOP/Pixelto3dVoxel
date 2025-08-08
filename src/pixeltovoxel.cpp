#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include "camera.h"
#include "math_utils.h"
#include "voxel_grid.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// --- forward declarations --------------------------------------------------
static void load_metadata(const std::string& path, std::vector<Camera>& cams, VoxelGrid& grid);
static void cast_ray(const cv::Vec3f& camPos, const cv::Vec3f& dir, VoxelGrid& grid, float pixVal);

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: pixeltovoxel <metadata.json>" << std::endl;
        return 1;
    }
    std::vector<Camera> cams;
    VoxelGrid grid(1,1.0f,{0,0,0}); // temp, will be replaced in load
    load_metadata(argv[1], cams, grid);

    fs::create_directories("build");

    const float MOTION_THRESHOLD = 2.0f;
    const float THRESH = 5.0f;
    const float half = 0.5f * grid.N * grid.voxelSize;
    const cv::Vec3f boxMin = grid.center - cv::Vec3f(half, half, half);
    size_t maxFrames = 0;
    for (const auto& c : cams) maxFrames = std::max(maxFrames, c.frames.size());

    std::vector<cv::Mat> prevImgs(cams.size());
    for (size_t fi = 0; fi < maxFrames; ++fi) {
        std::ostringstream oss;
        oss << "build/hits_" << std::setw(4) << std::setfill('0') << fi << ".xyz";
        std::ofstream xyz(oss.str());

        for (size_t ci = 0; ci < cams.size(); ++ci) {
            const Camera& cam = cams[ci];
            if (fi >= cam.frames.size()) continue;
            cv::Mat img = cv::imread(cam.frames[fi], cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;
            if (prevImgs[ci].empty()) { prevImgs[ci] = img; continue; }

            grid.clear();
            cv::Mat diff;
            cv::absdiff(img, prevImgs[ci], diff);
            prevImgs[ci] = img;

            // Focal length in pixels from horizontal FOV
            const float focal_px = 0.5f * diff.cols / tanf(0.5f * cam.fov_deg * CV_PI / 180.0f);
            const cv::Matx33f Rcw = cam.rotation();
            for (int v = 0; v < diff.rows; ++v) {
                const uchar* row = diff.ptr<uchar>(v);
                for (int u = 0; u < diff.cols; ++u) {
                    if (row[u] < MOTION_THRESHOLD) continue;
                    float px = (u - 0.5f * diff.cols);    // image right
                    float py = -(v - 0.5f * diff.rows);   // image up
                    float pz = focal_px;                 // forward

                    // Convert image coordinates to body frame:
                    // body X=fwd, Y=left, Z=up
                    cv::Vec3f dir_body(pz, -px, py);
                    cv::Vec3f dir_world = Rcw * cv::normalize(dir_body);
                    cast_ray(cam.position, dir_world, grid, static_cast<float>(row[u]));
                }
            }

            for (int iz=0; iz<grid.N; ++iz)
              for (int iy=0; iy<grid.N; ++iy)
                for (int ix=0; ix<grid.N; ++ix) {
                  float v = grid.data[(iz*grid.N + iy)*grid.N + ix];
                  if (v < THRESH) continue;
                  float x = boxMin[0] + (ix + 0.5f) * grid.voxelSize;
                  float y = boxMin[1] + (iy + 0.5f) * grid.voxelSize;
                  float z = boxMin[2] + (iz + 0.5f) * grid.voxelSize;
                  xyz << cam.id << " " << x << " " << y << " " << z << " " << v << "\n";
                }
        }
    }

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

    fs::path base = fs::path(path).parent_path();

    for (auto& jc : j["cameras"]) {
        Camera cam;
        cam.id      = jc["id"].get<std::string>();
        cam.folder  = (base / jc["folder"].get<std::string>()).string();
        cam.position = { jc["position"][0].get<float>(),
                         jc["position"][1].get<float>(),
                         jc["position"][2].get<float>() };
        cam.ypr_deg = { jc["yaw_pitch_roll"][0].get<float>(),
                        jc["yaw_pitch_roll"][1].get<float>(),
                        jc["yaw_pitch_roll"][2].get<float>() };
        cam.fov_deg = jc["fov_degrees"].get<float>();

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
    cv::Vec3f tDelta = cv::Vec3f(grid.voxelSize, grid.voxelSize, grid.voxelSize).mul(invDir);
    tDelta[0] = std::abs(tDelta[0]);
    tDelta[1] = std::abs(tDelta[1]);
    tDelta[2] = std::abs(tDelta[2]);

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
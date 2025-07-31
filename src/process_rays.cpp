#include <opencv2/core.hpp>
#include <fstream>
#include <vector>
#include <iostream>

int main() {
    int N; float voxelSize; cv::Vec3f center;
    std::ifstream in("rays.bin", std::ios::binary);
    if (!in) { std::cerr << "rays.bin not found"; return 1; }
    in.read(reinterpret_cast<char*>(&N), sizeof(int));
    in.read(reinterpret_cast<char*>(&voxelSize), sizeof(float));
    in.read(reinterpret_cast<char*>(&center), sizeof(float)*3);
    std::vector<float> grid(N*N*N);
    in.read(reinterpret_cast<char*>(grid.data()), sizeof(float)*grid.size());
    in.close();

    const float THRESH = 5.0f; // simple global threshold
    std::ofstream xyz("hits.xyz");
    for (int iz=0; iz<N; ++iz)
      for (int iy=0; iy<N; ++iy)
        for (int ix=0; ix<N; ++ix) {
          float v = grid[(iz*N + iy)*N + ix];
          if (v < THRESH) continue;
          float x = center[0] + (ix - N*0.5f + 0.5f)*voxelSize;
          float y = center[1] + (iy - N*0.5f + 0.5f)*voxelSize;
          float z = center[2] + (iz - N*0.5f + 0.5f)*voxelSize;
          xyz << x << " " << y << " " << z << " " << v << "\n";
        }
    xyz.close();
    return 0;
}
cmake_minimum_required(VERSION 3.15)
project(voxel_drone_tracking)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

# --- nlohmann/json header --------------------------------------------------
# Prefer system package (Homebrew / apt / vcpkg). Fallback to ./external/.
find_path(JSON_INCLUDE_DIR nlohmann/json.hpp
          HINTS /opt/homebrew/opt/nlohmann-json/include
                /usr/local/include
                /usr/include
                ${CMAKE_CURRENT_SOURCE_DIR}/external
          NO_DEFAULT_PATH)
if(NOT JSON_INCLUDE_DIR)
  message(FATAL_ERROR "nlohmann/json.hpp not found. Either:\n"
                      "  • brew install nlohmann-json  (or apt/vcpkg equivalent)\n"
                      "  • curl -L -o external/nlohmann/json.hpp "
                      "https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp")
endif()
include_directories(include ${JSON_INCLUDE_DIR})

# --------------------------------------------------------------------------
file(GLOB VOXEL_SOURCES src/*.cpp)
add_executable(pixeltovoxel src/pixeltovoxel.cpp)
add_executable(process_rays src/process_rays.cpp)

target_link_libraries(pixeltovoxel PRIVATE ${OpenCV_LIBS})
target_link_libraries(process_rays  PRIVATE ${OpenCV_LIBS})

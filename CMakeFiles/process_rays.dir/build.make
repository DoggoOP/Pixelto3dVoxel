# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel

# Include any dependencies generated for this target.
include CMakeFiles/process_rays.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/process_rays.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/process_rays.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/process_rays.dir/flags.make

CMakeFiles/process_rays.dir/codegen:
.PHONY : CMakeFiles/process_rays.dir/codegen

CMakeFiles/process_rays.dir/src/process_rays.cpp.o: CMakeFiles/process_rays.dir/flags.make
CMakeFiles/process_rays.dir/src/process_rays.cpp.o: src/process_rays.cpp
CMakeFiles/process_rays.dir/src/process_rays.cpp.o: CMakeFiles/process_rays.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/process_rays.dir/src/process_rays.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/process_rays.dir/src/process_rays.cpp.o -MF CMakeFiles/process_rays.dir/src/process_rays.cpp.o.d -o CMakeFiles/process_rays.dir/src/process_rays.cpp.o -c /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel/src/process_rays.cpp

CMakeFiles/process_rays.dir/src/process_rays.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/process_rays.dir/src/process_rays.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel/src/process_rays.cpp > CMakeFiles/process_rays.dir/src/process_rays.cpp.i

CMakeFiles/process_rays.dir/src/process_rays.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/process_rays.dir/src/process_rays.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel/src/process_rays.cpp -o CMakeFiles/process_rays.dir/src/process_rays.cpp.s

# Object files for target process_rays
process_rays_OBJECTS = \
"CMakeFiles/process_rays.dir/src/process_rays.cpp.o"

# External object files for target process_rays
process_rays_EXTERNAL_OBJECTS =

process_rays: CMakeFiles/process_rays.dir/src/process_rays.cpp.o
process_rays: CMakeFiles/process_rays.dir/build.make
process_rays: /opt/homebrew/lib/libopencv_gapi.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_stitching.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_alphamat.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_aruco.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_bgsegm.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_bioinspired.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_ccalib.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_dnn_objdetect.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_dnn_superres.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_dpm.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_face.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_freetype.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_fuzzy.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_hfs.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_img_hash.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_intensity_transform.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_line_descriptor.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_mcc.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_quality.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_rapid.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_reg.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_rgbd.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_saliency.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_sfm.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_signal.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_stereo.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_structured_light.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_superres.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_surface_matching.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_tracking.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_videostab.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_viz.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_wechat_qrcode.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_xfeatures2d.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_xobjdetect.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_xphoto.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_shape.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_highgui.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_datasets.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_plot.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_text.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_ml.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_phase_unwrapping.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_optflow.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_ximgproc.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_video.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_videoio.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_imgcodecs.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_objdetect.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_calib3d.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_dnn.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_features2d.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_flann.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_photo.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_imgproc.4.12.0.dylib
process_rays: /opt/homebrew/lib/libopencv_core.4.12.0.dylib
process_rays: CMakeFiles/process_rays.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable process_rays"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/process_rays.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/process_rays.dir/build: process_rays
.PHONY : CMakeFiles/process_rays.dir/build

CMakeFiles/process_rays.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/process_rays.dir/cmake_clean.cmake
.PHONY : CMakeFiles/process_rays.dir/clean

CMakeFiles/process_rays.dir/depend:
	cd /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel /Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel/CMakeFiles/process_rays.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/process_rays.dir/depend


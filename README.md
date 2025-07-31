# Pixelto3dVoxel

Pixelto3dVoxel implements a **pixel-to-voxel projection** pipeline for 3D object localization. The
approach fuses observations from multiple calibrated cameras to locate a moving target (e.g. a
drone) in a shared world coordinate system.

## Architecture Overview

1. **Input Calibration**: Each camera frame comes with metadata describing the camera position,
   orientation (yaw, pitch, roll) and field of view. This calibration allows the software to
   convert pixel coordinates into absolute 3D rays.

2. **Motion Detection**: Consecutive frames from every camera are compared to detect moving pixels.
   Simple frame differencing identifies pixels that changed significantly between frames.

3. **Ray Casting**: For each changed pixel, the code projects a 3D ray into the scene. The ray is
   expressed in world coordinates using the camera orientation, and then intersected with a 3D
   voxel grid representing the volume of interest.

4. **Voxel Accumulation**: Voxels along each ray accumulate the pixel difference value. When
   different cameras observe the same object, their projected rays intersect and the shared voxels
   receive higher values, indicating a potential object location.

5. **Output**: After processing all frames, the populated voxel grid is written to a binary file for
   further visualization or analysis.

This voxel-based accumulation approach is robust to noise and can fuse many camera views into a
single 3D occupancy map.


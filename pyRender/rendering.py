from pathlib import Path

import numpy as np
import pyvista as pv


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0=0.0):
    """Convert WGS-84 geodetic coordinates to local ENU (metres)."""
    R = 6_378_137.0
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    east = R * dlon * np.cos(np.deg2rad(lat0))
    north = R * dlat
    up = alt - alt0
    return np.array([east, north, up])


def ypr_matrix(yaw_deg, pitch_deg, roll_deg):
    """Rotation matrix for yaw, pitch, roll in degrees."""
    yaw, pitch, roll = map(np.deg2rad, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Path to a 3D model file (.obj, .stl, .glb).
# Set to None to use PyVista's example airplane.
MODEL_PATH = "/Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel/pyRender/lancet-uav.obj"

# Camera position in latitude, longitude and altitude (metres).
CAM_LAT = 0
CAM_LON = 0
CAM_ALT = 1.7

# Camera orientation offsets in degrees.
CAM_YAW = 0.0
CAM_PITCH = 5.0
CAM_ROLL = 0.0

# Plane start and end positions in latitude, longitude and altitude (metres).
PLANE_START_LAT = -0.01
PLANE_START_LON = -0.01
PLANE_START_ALT = 10.0

PLANE_END_LAT = -0.01
PLANE_END_LON = 0.01
PLANE_END_ALT = 10.0

# Video parameters.
DURATION = 5.0  # seconds
FRAMERATE = 30  # frames per second
OUT_DIR = Path("frames")


# -----------------------------------------------------------------------------
# Main rendering
# -----------------------------------------------------------------------------

def main():
    camera_position = geodetic_to_enu(CAM_LAT, CAM_LON, CAM_ALT, CAM_LAT, CAM_LON, CAM_ALT)

    start_pos = geodetic_to_enu(
        PLANE_START_LAT,
        PLANE_START_LON,
        PLANE_START_ALT,
        CAM_LAT,
        CAM_LON,
        CAM_ALT,
    )
    end_pos = geodetic_to_enu(
        PLANE_END_LAT,
        PLANE_END_LON,
        PLANE_END_ALT,
        CAM_LAT,
        CAM_LON,
        CAM_ALT,
    )

    num_frames = max(1, int(DURATION * FRAMERATE))
    plane_path = np.linspace(start_pos, end_pos, num_frames)

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1088])

    model = pv.read(MODEL_PATH) if MODEL_PATH else pv.examples.load_airplane()
    model.points += start_pos
    last_pos = start_pos.copy()

    for i, pos in enumerate(plane_path):
        plotter.clear()
        plotter.enable_lightkit()
        plotter.set_background("skyblue")

        model.points += pos - last_pos
        last_pos = pos
        plotter.add_mesh(model, color="silver")

        ground = pv.Plane(center=(0, -500, 0), direction=(0, 0, 1), i_size=4000, j_size=4000)
        plotter.add_mesh(ground, color="#46613a")

        forward_vec = pos - camera_position
        dist = np.linalg.norm(forward_vec)
        if dist == 0:
            forward_vec = np.array([1.0, 0.0, 0.0])
            dist = 1.0
        forward_dir = forward_vec / dist
        R = ypr_matrix(CAM_YAW, CAM_PITCH, CAM_ROLL)
        forward_rot = R @ forward_dir
        view_up = R @ np.array([0.0, 0.0, 1.0])
        focal_point = camera_position + forward_rot * dist

        plotter.camera_position = [camera_position, focal_point, view_up]
        plotter.add_light(pv.Light(position=(10, 10, 1000), light_type="scene light"))

        frame_file = out_dir / f"frame_{i:04d}.png"
        plotter.screenshot(str(frame_file), window_size=[1920, 1088])
        print(f"Saved {frame_file}", end="\r")

    plotter.close()
    print("\nAll frames saved →", out_dir)


if __name__ == "__main__":
    main()
"""
UAV fly‑by renderer (PyVista)
──────────────────────────
• Geodetic lat/long → local ENU metres (x=east, y=north, z=up).
• Camera yaw = 0 deg = **true north**, yaw increases clockwise (90 deg = east).
• Pitch +ve = look‑up, roll +ve = right‑wing‑down — all in degrees.
• Produces PNG frames under ./frames/
"""

from pathlib import Path
import numpy as np
import pyvista as pv

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0=0.0):
    """WGS‑84 → local ENU (m). Camera lat/lon is the local origin."""
    R = 6_378_137.0  # mean Earth radius (m)
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    east  = R * dlon * np.cos(np.deg2rad(lat0))
    north = R * dlat
    up    = alt - alt0
    return np.array([east, north, up])


def ypr_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float):
    """Rotation Z‑Y‑X with yaw = 0°→North, +cw.
    Returns 3×3 DCM body←world."""
    
    yaw = np.deg2rad(90-yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    cy, sy = np.cos(yaw),  np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)

    Rz = np.array([[ cy, -sy, 0],
                   [ sy,  cy, 0],
                   [  0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1, 0],
                   [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx  # body ← world

# ────────────────────────────────────────────────────────────────────────────────
# Configuration (edit as needed)
# ────────────────────────────────────────────────────────────────────────────────

MODEL_PATH = (
    "/Users/ethan.k/Desktop/CodingNStuff/python/Pixelto3dVoxel/pyRender/"
    "lancet-uav.obj"
)

# Camera geodetic position (deg, deg, m)
CAM_LAT, CAM_LON, CAM_ALT = 0.05, 0, 1.7
# Camera attitude (deg) — yaw 0 = north
CAM_YAW, CAM_PITCH, CAM_ROLL = 180, 5.0, 0.0

# UAV path endpoints (deg, deg, m)
PLANE_START_LAT, PLANE_START_LON, PLANE_START_ALT = 0.0, -0.05, 10.0
PLANE_END_LAT,   PLANE_END_LON,   PLANE_END_ALT   = 0.0,  0.05, 10.0

# Output / timing
DURATION   = 5.0   # s
FRAMERATE  = 30    # Hz
OUT_DIR    = Path("frames")
WINDOWSIZE = [1920, 1088]

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # Local origin at camera
    cam_pos = geodetic_to_enu(CAM_LAT, CAM_LON, CAM_ALT,
                              CAM_LAT, CAM_LON, CAM_ALT)

    start_pos = geodetic_to_enu(PLANE_START_LAT, PLANE_START_LON, PLANE_START_ALT,
                                CAM_LAT, CAM_LON, CAM_ALT)
    end_pos   = geodetic_to_enu(PLANE_END_LAT,   PLANE_END_LON,   PLANE_END_ALT,
                                CAM_LAT, CAM_LON, CAM_ALT)

    num_frames = max(1, int(DURATION * FRAMERATE))
    plane_path = np.linspace(start_pos, end_pos, num_frames)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(off_screen=True, window_size=WINDOWSIZE)

    model = pv.read(MODEL_PATH) if MODEL_PATH else pv.examples.load_airplane()
    model.points += start_pos
    last_pos = start_pos.copy()

    # Camera orientation
    R_cam = ypr_matrix(CAM_YAW, CAM_PITCH, CAM_ROLL)
    forward_vec = R_cam @ np.array([1.0, 0.0, 0.0])  # body +X
    view_up = R_cam @ np.array([0.0, 0.0, 1.0])      # body +Z
    view_dist = 1000.0  # m ahead

    for i, pos in enumerate(plane_path):
        plotter.clear()
        plotter.enable_lightkit()
        plotter.set_background("skyblue")

        # Move UAV
        model.points += pos - last_pos
        last_pos = pos
        plotter.add_mesh(model, color="silver")

        # Simple ground
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                          i_size=8000, j_size=8000)
        plotter.add_mesh(ground, color="#46613a")

        # Camera
        focal_point = cam_pos + forward_vec * view_dist
        plotter.camera_position = [cam_pos, focal_point, view_up]

        # Lighting
        plotter.add_light(pv.Light(position=(10_000, 10_000, 30_000),
                                   light_type="scene light"))

        frame_file = OUT_DIR / f"frame_{i:04d}.png"
        plotter.screenshot(str(frame_file), window_size=WINDOWSIZE)
        print(f"Saved {frame_file}", end="\r")

    plotter.close()
    print("\nAll frames saved →", OUT_DIR)


if __name__ == "__main__":
    main()

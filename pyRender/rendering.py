import pyvista as pv
import numpy as np
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0=0.0):
    """Convert WGS‑84 geodetic coordinates to local ENU (metres)."""
    R = 6_378_137.0
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    east  = R * dlon * np.cos(np.deg2rad(lat0))
    north = R * dlat
    up    = alt - alt0
    return np.array([east, north, up])


def ypr_to_forward_up(yaw_deg, pitch_deg, roll_deg):
    """Return forward & up unit‑vectors for yaw/pitch/roll (deg)."""
    yaw, pitch, roll = map(np.deg2rad, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = np.cos(yaw),  np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[ cy, -sy, 0],
                   [ sy,  cy, 0],
                   [  0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1, 0],
                   [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    R = Rz @ Ry @ Rx
    fwd = R @ np.array([1, 0, 0])
    up  = R @ np.array([0, 0, 1])
    return fwd/np.linalg.norm(fwd), up/np.linalg.norm(up)

# ────────────────────────────────────────────────────────────────────────────────
# Scene parameters
# ────────────────────────────────────────────────────────────────────────────────

cam_lat, cam_lon, cam_alt = 38.7260, -9.1406, 1.7      # metres above WGS-84
cam_yaw, cam_pitch, cam_roll = 60.0, 5.0, 0.0         # °  (+pitch = look-up)

camera_position = geodetic_to_enu(cam_lat, cam_lon, cam_alt,
                                  cam_lat, cam_lon)
forward, view_up = ypr_to_forward_up(cam_yaw, cam_pitch, cam_roll)

# Aim 15 km along that axis (change if the plane is farther/closer)
focal_point_static = camera_position + forward * 15_000

distance = 1000
start_pos = np.array([12 * distance,  5 * distance, 6 * distance])
end_pos   = np.array([12 * distance, -5 * distance, 6 * distance])

video_duration, framerate = 5, 30
num_frames = video_duration * framerate
plane_path = np.linspace(start_pos, end_pos, num_frames)

# ────────────────────────────────────────────────────────────────────────────────
# Output directory for frame images
# ────────────────────────────────────────────────────────────────────────────────

out_dir = Path("frames")
out_dir.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# PyVista render loop
# ────────────────────────────────────────────────────────────────────────────────

plotter = pv.Plotter(off_screen=True, window_size=[1920, 1088])

airplane = pv.examples.load_airplane()
airplane.points += start_pos
last_pos = start_pos.copy()

for i, pos in enumerate(plane_path):
    plotter.clear()
    plotter.enable_lightkit()
    plotter.set_background("skyblue")

    # Move airplane mesh
    airplane.points += pos - last_pos
    last_pos = pos
    plotter.add_mesh(airplane, color='silver')

    # Ground plane
    ground = pv.Plane(center=(0, -500, 0), direction=(0, 0, 1),
                      i_size=4000, j_size=4000)
    plotter.add_mesh(ground, color='#46613a')

    # Camera setup
    plotter.camera_position = [camera_position,
                               focal_point_static,
                               view_up]

    # Lighting
    plotter.add_light(pv.Light(position=(10, 10, 1000), light_type='scene light'))

    # Write frame as PNG
    frame_file = out_dir / f"frame_{i:04d}.png"
    plotter.screenshot(str(frame_file), window_size=[1920, 1088])
    print(f"Saved {frame_file}", end="\r")

plotter.close()
print("\nAll frames saved →", out_dir)

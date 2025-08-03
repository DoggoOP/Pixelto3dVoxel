import argparse
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


def ypr_to_forward_up(yaw_deg, pitch_deg, roll_deg):
    """Return forward & up unit-vectors for yaw/pitch/roll (deg)."""
    yaw, pitch, roll = map(np.deg2rad, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R = Rz @ Ry @ Rx
    fwd = R @ np.array([1, 0, 0])
    up = R @ np.array([0, 0, 1])
    return fwd / np.linalg.norm(fwd), up / np.linalg.norm(up)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    """CLI arguments for camera placement and model path."""
    parser = argparse.ArgumentParser(
        description="Render a flying model with a custom camera."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to 3D model file (.obj, .stl, .glb). Defaults to PyVista's example airplane.",
    )
    parser.add_argument("--cam-lat", type=float, default=38.7260, help="Camera latitude in degrees.")
    parser.add_argument("--cam-lon", type=float, default=-9.1406, help="Camera longitude in degrees.")
    parser.add_argument("--cam-alt", type=float, default=1.7, help="Camera altitude in metres.")
    parser.add_argument("--cam-yaw", type=float, default=60.0, help="Camera yaw in degrees.")
    parser.add_argument("--cam-pitch", type=float, default=5.0, help="Camera pitch in degrees.")
    parser.add_argument("--cam-roll", type=float, default=0.0, help="Camera roll in degrees.")
    parser.add_argument("--duration", type=float, default=5.0, help="Video duration in seconds.")
    parser.add_argument("--framerate", type=int, default=30, help="Frames per second.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("frames"),
        help="Directory to save output frames.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main rendering
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    camera_position = geodetic_to_enu(
        args.cam_lat, args.cam_lon, args.cam_alt, args.cam_lat, args.cam_lon
    )
    forward, view_up = ypr_to_forward_up(args.cam_yaw, args.cam_pitch, args.cam_roll)
    focal_point_static = camera_position + forward * 15_000

    distance = 1000
    start_pos = np.array([12 * distance, 5 * distance, 6 * distance])
    end_pos = np.array([12 * distance, -5 * distance, 6 * distance])

    num_frames = int(args.duration * args.framerate)
    plane_path = np.linspace(start_pos, end_pos, num_frames)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1088])

    model = pv.read(args.model) if args.model else pv.examples.load_airplane()
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

        plotter.camera_position = [camera_position, focal_point_static, view_up]
        plotter.add_light(pv.Light(position=(10, 10, 1000), light_type="scene light"))

        frame_file = out_dir / f"frame_{i:04d}.png"
        plotter.screenshot(str(frame_file), window_size=[1920, 1088])
        print(f"Saved {frame_file}", end="\r")

    plotter.close()
    print("\nAll frames saved →", out_dir)


if __name__ == "__main__":
    main()

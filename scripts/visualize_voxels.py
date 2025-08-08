#!/usr/bin/env python3
"""
Animate the per-frame voxel hits together with the camera rays.

Inputs
------
build/hits_<camid>_*.xyz   text files:  x  y  z  value
metadata.json              camera calibration with "position"

Run
---
python scripts/visualize_voxels.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from vedo import Lines, Points, Plotter, Box, Sphere, Text3D, settings

# ------------------------------------------------------------------ helpers
def _maybe_start_xvfb() -> None:
    """Start an off-screen OpenGL buffer only on headless Linux boxes."""
    if sys.platform.startswith("linux") and not sys.stdout.isatty():
        settings.start_xvfb()


def _load_meta(meta_file: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Return camera positions, ids and voxel grid bounds from metadata."""
    import json

    with meta_file.open() as f:
        meta = json.load(f)
    cams = meta["cameras"]
    cam_ids = [c["id"] for c in cams]
    cam_pos = np.asarray([c["position"] for c in cams], dtype=np.float32)

    v = meta["voxel"]
    N = float(v["N"]) * v["voxel_size"]
    center = np.asarray(v["center"], dtype=np.float32)
    half = 0.5 * N
    box_min, box_max = center - half, center + half
    return cam_pos, cam_ids, box_min, box_max


# ------------------------------------------------------------------ main
def main() -> None:
    _maybe_start_xvfb()

    ROOT = Path(__file__).resolve().parent.parent
    BUILD = ROOT / "build"
    cam_pos, cam_ids, bmin, bmax = _load_meta(ROOT / "metadata.json")
    lat_vals = np.round(np.arange(bmin[1], bmax[1]+1e-6, 0.001), 3)
    lon_vals = np.round(np.arange(bmin[0], bmax[0]+1e-6, 0.001), 3)
    axes_opts = dict(xrange=(bmin[0], bmax[0]),
                     yrange=(bmin[1], bmax[1]),
                     zrange=(bmin[2], bmax[2]),
                     xtitle="longitude",
                     ytitle="latitude",
                     ztitle="height",
                     x_values_and_labels=[(v, f"{v:.3f}") for v in lon_vals],
                     y_values_and_labels=[(v, f"{v:.3f}") for v in lat_vals])

    # actors created once ----------------------------------------------------
    colors = ["red", "green", "blue", "magenta", "orange", "cyan"]
    cam_actors = [Sphere(pos=cam, r=0.1, c=colors[i % len(colors)])
                  for i, cam in enumerate(cam_pos)]
    cam_labels = [Text3D(cid, cam + np.array([0.2, 0.2, 0]), s=8,
                         c=colors[i % len(colors)])
                  for i, (cam, cid) in enumerate(zip(cam_pos, cam_ids))]

    pts_actors = [Points([[0, 0, 0]], r=4, c=colors[i % len(colors)])
                  for i in range(len(cam_ids))]
    ray_lines = [Lines([cam], [cam], c=colors[i % len(colors)], lw=1)
                 for i, cam in enumerate(cam_pos)]

    grid_box = Box(pos=(bmin + bmax) / 2, size=bmax - bmin,
                   c=None, alpha=0.1).wireframe()

    plt = Plotter(bg="white", axes=axes_opts, interactive=False,
                  title="Voxel hits with camera rays")
    plt += [grid_box, *cam_actors, *cam_labels, *pts_actors, *ray_lines]
    plt.show(resetcam=True, viewup="z", azimuth=45, elevation=-45)

    # gather xyz files -------------------------------------------------------
    xyz_files = sorted(BUILD.glob("hits_*_*.xyz"))
    if not xyz_files:
        sys.exit("No hits files found in build/")

    frame_ids = sorted({int(f.stem.split("_")[-1]) for f in xyz_files})

    # frame loop -------------------------------------------------------------
    for fi in frame_ids:
        for ci, cid in enumerate(cam_ids):
            xyz = BUILD / f"hits_{cid}_{fi:04d}.xyz"
            if not xyz.exists() or xyz.stat().st_size == 0:
                ray_lines[ci].points = [cam_pos[ci], cam_pos[ci]]
                pts_actors[ci].points = []
                continue

            a = np.loadtxt(xyz, dtype=np.float64)
            if a.ndim == 1:
                a = a[None]

            coords, vals = a[:, :3], a[:, 3]

            pts_actors[ci].points = coords
            pts_actors[ci].pointdata["val"] = vals
            pts_actors[ci].cmap("jet", "val")

            n = len(coords)
            ps = np.empty((2 * n, 3), dtype=np.float32)
            ps[0::2] = cam_pos[ci]
            ps[1::2] = coords
            ray_lines[ci].points = ps

        plt.render()
        time.sleep(0.03)                            # ~30 fps

    plt.close()


if __name__ == "__main__":
    main()

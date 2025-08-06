#!/usr/bin/env python3
"""
Animate the per-frame voxel hits together with the 3 camera rays.

Inputs
------
build/hits_*.xyz        text files:  x  y  z  value
metadata.json           camera calibration with "position"

Run
---
python scripts/visualize_voxels.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from vedo import Lines, Points, Plotter, Box, settings

# ------------------------------------------------------------------ helpers
def _maybe_start_xvfb() -> None:
    """Start an off-screen OpenGL buffer only on headless Linux boxes."""
    if sys.platform.startswith("linux") and not sys.stdout.isatty():
        settings.start_xvfb()


def _load_meta(meta_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera positions and voxel grid bounds from metadata."""
    import json

    with meta_file.open() as f:
        meta = json.load(f)
    cam_pos = np.asarray([cam["position"] for cam in meta["cameras"]],
                         dtype=np.float32)

    v = meta["voxel"]
    N = float(v["N"]) * v["voxel_size"]
    center = np.asarray(v["center"], dtype=np.float32)
    half = 0.5 * N
    box_min, box_max = center - half, center + half
    return cam_pos, box_min, box_max


# ------------------------------------------------------------------ main
def main() -> None:
    _maybe_start_xvfb()

    ROOT = Path(__file__).resolve().parent.parent
    BUILD = ROOT / "build"
    cam_pos, bmin, bmax = _load_meta(ROOT / "metadata.json")
    axes_opts = dict(xrange=(bmin[0], bmax[0]),
                     yrange=(bmin[1], bmax[1]),
                     zrange=(bmin[2], bmax[2]))

    # actors created once ----------------------------------------------------
    cam_actor = Points(cam_pos, r=12, c="black")

    pts_actor = Points([[0, 0, 0]], r=4)           # placeholder
    ray_lines = [Lines([cam], [cam], c="black", lw=1) for cam in cam_pos]

    grid_box = Box(pos=(bmin + bmax) / 2, size=bmax - bmin,
                   c=None, alpha=0.1).wireframe()

    plt = Plotter(bg="white", axes=axes_opts, interactive=False,
                  title="Voxel hits with camera rays")
    plt += [pts_actor, cam_actor, grid_box, *ray_lines]
    plt.show(resetcam=True, viewup="z", azimuth=45, elevation=-45)

    # gather xyz files -------------------------------------------------------
    xyz_files = sorted(BUILD.glob("hits_*.xyz"))
    if not xyz_files:
        sys.exit("No hits_*.xyz files found in build/")

    # frame loop -------------------------------------------------------------
    for xyz in xyz_files:
        if xyz.stat().st_size == 0:
            continue

        a = np.loadtxt(xyz, dtype=np.float64)       # (N,4) or (4,)
        if a.ndim == 1:
            a = a[None]                             # single point → (1,4)

        coords, vals = a[:, :3], a[:, 3]            # split xyz + value

        # update points – copy=True ⇒ vtk owns its own buffer
        pts_actor.points = coords        # vedo ≥ 2024.5 :contentReference[oaicite:0]{index=0}
        pts_actor.pointdata["val"] = vals
        pts_actor.cmap("jet", "val")

        # draw rays from each camera to every hit
        for cam, lines in zip(cam_pos, ray_lines):
            n = len(coords)
            ps = np.empty((2 * n, 3), dtype=np.float32)
            ps[0::2] = cam
            ps[1::2] = coords
            lines.points = ps

        plt.render()
        time.sleep(0.03)                            # ~30 fps

    plt.close()


if __name__ == "__main__":
    main()

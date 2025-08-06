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
from vedo import Line, Points, Plotter, settings


# ------------------------------------------------------------------ helpers
def _maybe_start_xvfb() -> None:
    """Start an off-screen OpenGL buffer only on headless Linux boxes."""
    if sys.platform.startswith("linux") and not sys.stdout.isatty():
        settings.start_xvfb(False)


def _load_camera_positions(meta_file: Path) -> np.ndarray:
    import json

    with meta_file.open() as f:
        meta = json.load(f)
    return np.asarray([cam["position"] for cam in meta["cameras"]], dtype=np.float32)


# ------------------------------------------------------------------ main
def main() -> None:
    _maybe_start_xvfb()

    ROOT = Path(__file__).resolve().parent.parent
    BUILD = ROOT / "build"
    cam_pos = _load_camera_positions(ROOT / "metadata.json")

    # scene limits
    bmin, bmax = cam_pos.min(0), cam_pos.max(0)
    axes_opts = dict(xrange=(bmin[0], bmax[0]),
                     yrange=(bmin[1], bmax[1]),
                     zrange=(bmin[2], bmax[2]))

    # actors created once ----------------------------------------------------
    cam_actor = Points(cam_pos, r=12, c="black")

    pts_actor = Points([[0, 0, 0]], r=4)           # placeholder
    ray_lines = [Line(cam_pos[i], cam_pos[i], c="black", lw=1)
                 for i in range(len(cam_pos))]

    plt = Plotter(bg="white", axes=axes_opts, interactive=False,
                  title="Voxel hits with camera rays")
    plt += [pts_actor, cam_actor, *ray_lines]
    plt.show(resetcam=True)                         # first render

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

        # update rays
        for i, line in enumerate(ray_lines):
            tgt = coords[i] if i < len(coords) else cam_pos[i]
            line.points = [cam_pos[i], tgt]

        plt.render()
        time.sleep(0.03)                            # ~30 fps

    plt.close()


if __name__ == "__main__":
    main()

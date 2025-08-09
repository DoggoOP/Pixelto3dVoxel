#!/usr/bin/env python3
"""
Animate the per-frame voxel hits together with the camera rays.

Inputs
------
build/hits_*.xyz        text files:  x  y  z  value  camMask
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
    tick = 0.005  # degrees
    xticks = np.arange(bmin[0], bmax[0] + tick, tick)
    yticks = np.arange(bmin[1], bmax[1] + tick, tick)
    if len(xticks) > 200:
        xticks = np.linspace(bmin[0], bmax[0], 11)
    if len(yticks) > 200:
        yticks = np.linspace(bmin[1], bmax[1], 11)
    zticks = np.linspace(bmin[2], bmax[2], 11)
    axes_opts = dict(xrange=(bmin[0], bmax[0]),
                     yrange=(bmin[1], bmax[1]),
                     zrange=(bmin[2], bmax[2]),
                     xtitle="lon (deg)",
                     ytitle="lat (deg)",
                     ztitle="alt (m)",
                     xticks=xticks,
                     yticks=yticks,
                     zticks=zticks)

    # actors created once ----------------------------------------------------
    cam_colors = ["red", "green", "blue", "orange", "purple", "cyan"]
    cam_actors = [Sphere(pos=cam, r=0.2, c=cam_colors[i % len(cam_colors)])
                  for i, cam in enumerate(cam_pos)]
    cam_labels = [Text3D(cid, cam + np.array([0.2, 0.2, 0]), s=8,
                         c=cam_colors[i % len(cam_colors)])
                  for i, (cam, cid) in enumerate(zip(cam_pos, cam_ids))]

    pts_actor = Points([[0, 0, 0]], r=4)           # placeholder
    ray_lines = [Lines([cam], [cam], c=cam_colors[i % len(cam_colors)], lw=1)
                 for i, cam in enumerate(cam_pos)]

    grid_box = Box(pos=(bmin + bmax) / 2, size=bmax - bmin,
                   c=None, alpha=0.1).wireframe()

    plt = Plotter(bg="white", axes=axes_opts, interactive=False,
                  title="Voxel hits with camera rays")
    plt += [pts_actor, grid_box, *cam_actors, *cam_labels, *ray_lines]
    try:
        plt.show(resetcam=True, viewup="z", azimuth=45, elevation=-45)
    except TypeError as e:
        # Older versions of vedo do not support explicit tick placement.
        if "ticks" in str(e):
            for k in ("xticks", "yticks", "zticks"):
                axes_opts.pop(k, None)
            plt.axes = axes_opts
            plt.show(resetcam=True, viewup="z", azimuth=45, elevation=-45)
        else:
            raise
    
    # gather xyz files -------------------------------------------------------
    xyz_files = sorted(BUILD.glob("hits_*.xyz"))
    if not xyz_files:
        sys.exit("No hits_*.xyz files found in build/")

    # frame loop -------------------------------------------------------------
    for xyz in xyz_files:
        if xyz.stat().st_size == 0:
            continue

        a = np.loadtxt(xyz, dtype=np.float64)       # (N,5) or (5,)
        if a.ndim == 1:
            a = a[None]

        coords, vals = a[:, :3], a[:, 3]
        masks = a[:, 4].astype(np.uint8)

        # update points – copy=True ⇒ vtk owns its own buffer
        pts_actor.points = coords        # vedo ≥ 2024.5
        pts_actor.pointdata["val"] = vals
        pts_actor.cmap("jet", "val")

        # draw rays from each camera only to its hits
        for ci, (cam, lines) in enumerate(zip(cam_pos, ray_lines)):
            mask = (masks & (1 << ci)) != 0
            pts = coords[mask]
            n = len(pts)
            if n == 0:
                lines.points = np.array([cam, cam])
                continue
            ps = np.empty((2 * n, 3), dtype=np.float32)
            ps[0::2] = cam
            ps[1::2] = pts
            lines.points = ps

        plt.render()
        time.sleep(0.03)                            # ~30 fps

    plt.close()


if __name__ == "__main__":
    main()

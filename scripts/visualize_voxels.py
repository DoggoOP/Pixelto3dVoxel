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
from vedo import Lines, Points, Plotter, Box, Sphere, Text3D, settings


def geodetic_to_enu(lat: float, lon: float, alt: float,
                    lat0: float, lon0: float, alt0: float = 0.0) -> np.ndarray:
    """WGS‑84 → local ENU metres."""
    R = 6_378_137.0
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    east = R * dlon * np.cos(np.deg2rad(lat0))
    north = R * dlat
    up = alt - alt0
    return np.array([east, north, up], dtype=np.float32)


def enu_to_geodetic(enu: np.ndarray, lat0: float, lon0: float, alt0: float = 0.0) -> np.ndarray:
    """Local ENU metres → WGS‑84 (lat, lon in degrees, alt in metres)."""
    R = 6_378_137.0
    east, north, up = enu[..., 0], enu[..., 1], enu[..., 2]
    lat = lat0 + np.rad2deg(north / R)
    lon = lon0 + np.rad2deg(east / (R * np.cos(np.deg2rad(lat0))))
    alt = alt0 + up
    return np.stack([lat, lon, alt], axis=-1)

# ------------------------------------------------------------------ helpers
def _maybe_start_xvfb() -> None:
    """Start an off-screen OpenGL buffer only on headless Linux boxes."""
    if sys.platform.startswith("linux") and not sys.stdout.isatty():
        settings.start_xvfb()


def _load_meta(meta_file: Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, bool, tuple[float, float, float]]:
    """Return camera positions, ids, voxel bounds, and geodetic reference."""
    import json

    with meta_file.open() as f:
        meta = json.load(f)
    cams = meta["cameras"]
    cam_ids = [c["id"] for c in cams]

    
    lat0 = lon0 = alt0 = 0.0
    use_geo = False
    if "geo_reference" in meta:
        gref = meta["geo_reference"]
        lat0 = gref["lat_deg"]
        lon0 = gref["lon_deg"]
        alt0 = gref.get("alt_m", 0.0)
        use_geo = True

    # metres ↔ degrees conversion factors
    R = 6_378_137.0
    deg_per_m = 180.0 / (np.pi * R)
    lon_factor = deg_per_m / np.cos(np.deg2rad(lat0)) if use_geo else 1.0
    lat_factor = deg_per_m if use_geo else 1.0

    cam_pos = []
    if use_geo:
        for c in cams:
            # x=longitude, y=latitude, z=altitude
            cam_pos.append([c["lon_deg"], c["lat_deg"], c.get("alt_m", 0.0)])
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
    else:
        for c in cams:
            cam_pos.append(c["position"])
        cam_pos = np.asarray(cam_pos, dtype=np.float32)

    v = meta["voxel"]
    half_m = 0.5 * float(v["N"]) * v["voxel_size"]
    if use_geo and "center_geo" in v:
        cg = v["center_geo"]
        center_enu = geodetic_to_enu(cg["lat_deg"], cg["lon_deg"], cg.get("alt_m", 0.0),
                                     lat0, lon0, alt0)
        half = 0.5 * N
        bmin = enu_to_geodetic(center_enu - half, lat0, lon0, alt0)
        bmax = enu_to_geodetic(center_enu + half, lat0, lon0, alt0)
        # reorder to (lon, lat, alt) for plotting
        bmin = bmin[[1, 0, 2]]
        bmax = bmax[[1, 0, 2]]
    else:
        center = np.asarray(v["center"], dtype=np.float32)
        half = 0.5 * N
        bmin, bmax = center - half, center + half
    return cam_pos, cam_ids, bmin, bmax, use_geo, (lat0, lon0, alt0)


# ------------------------------------------------------------------ main
def main() -> None:
    _maybe_start_xvfb()

    ROOT = Path(__file__).resolve().parent.parent
    BUILD = ROOT / "build"
    cam_pos, cam_ids, bmin, bmax, use_geo, ref = _load_meta(ROOT / "metadata.json")
    lat0, lon0, alt0 = ref
    axes_opts = dict(xrange=(bmin[0], bmax[0]),
                     yrange=(bmin[1], bmax[1]),
                     zrange=(bmin[2], bmax[2]),
                     xtitle="lon_deg", ytitle="lat_deg", ztitle="alt_m")

    # actors created once ----------------------------------------------------
    cam_actors = [Sphere(pos=cam, r=(0.001 if not use_geo else 1e-3), c="red") for cam in cam_pos]
    offset = np.array([0.2, 0.2, 0]) if not use_geo else np.array([2e-3, 2e-3, 0])
    cam_labels = [Text3D(cid, cam + offset, s=8, c="red")
                  for cam, cid in zip(cam_pos, cam_ids)]

    pts_actor = Points([[0, 0, 0]], r=0.4)           # placeholder
    ray_lines = [Lines([cam], [cam], c="black", lw=1) for cam in cam_pos]

    grid_box = Box(pos=(bmin + bmax) / 2, size=bmax - bmin,
                   c=None, alpha=0.1).wireframe()

    plt = Plotter(bg="white", axes=axes_opts, interactive=False,
                  title="Voxel hits with camera rays")
    plt += [pts_actor, grid_box, *cam_actors, *cam_labels, *ray_lines]
    plt.show(resetcam=True, viewup="z", azimuth=45, elevation=-45,
         zoom=2.0)

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
        if use_geo:
            coords = enu_to_geodetic(coords, lat0, lon0, alt0)
            coords = coords[:, [1, 0, 2]]

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

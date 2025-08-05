from vedo import Points, show, settings
import numpy as np, os, glob, json, time

ROOT = os.path.dirname(__file__)
BUILD = os.path.join(ROOT, '..', 'build')

# Load camera positions to define bounds
with open(os.path.join(ROOT, '..', 'metadata.json')) as f:
    meta = json.load(f)
cam_pos = np.array([c['position'] for c in meta['cameras']])
bmin = cam_pos.min(axis=0)
bmax = cam_pos.max(axis=0)
axes_opts = dict(xrange=(bmin[0], bmax[0]),
                 yrange=(bmin[1], bmax[1]),
                 zrange=(bmin[2], bmax[2]))
cam_actor = Points(cam_pos, r=12, c='black')

files = sorted(glob.glob(os.path.join(BUILD, 'hits_*.xyz')))
if not files:
    raise SystemExit('No hits_*.xyz files found')

plot = None
pts_actor = None
for f in files:
    if os.path.getsize(f) == 0:
        continue
    pts = np.loadtxt(f)
    # Ensure pts is a 2D array even when a single point is present
    if pts.ndim == 1:
        pts = pts[None, :]
    # copy to ensure vedo does not reference temporary memory that may be freed
    coords = pts[:, :3].copy()
    vals = pts[:, 3].copy()
    if pts_actor is None:
        pts_actor = Points(coords, r=4)
        # stash arrays on the actor so Python keeps them alive
        pts_actor._coords_ref = coords
        pts_actor._vals_ref = vals
        pts_actor.pointdata["val"] = vals
        pts_actor.cmap('jet', "val")
        plot = show([pts_actor, cam_actor], axes=axes_opts, bg='white',
                    interactive=False, title='Voxel Hits')
    else:
        pts_actor.points = coords
        pts_actor._coords_ref = coords
        pts_actor.pointdata["val"] = vals
        pts_actor._vals_ref = vals
        pts_actor.cmap('jet', "val")
        plot.show(None, resetcam=False)
    time.sleep(0.1)

if plot is not None:
    plot.close()
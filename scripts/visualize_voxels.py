from vedo import Points, show, settings
import numpy as np, os, time, sys

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
    pts = np.loadtxt(f)
    if pts.size == 0:
        continue
    coords, vals = pts[:, :3], pts[:, 3]
    if pts_actor is None:
        pts_actor = Points(coords, r=4).cmap('jet', vals)
        plot = show([pts_actor, cam_actor], axes=axes_opts, bg='white',
                    interactive=False, title='Voxel Hits')
    else:
        pts_actor.points(coords)
        pts_actor.cmap('jet', vals)
        plot.show(None, resetcam=False)
    time.sleep(0.1)


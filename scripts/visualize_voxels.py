from vedo import Line, Points, show, settings
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
ray_actors = []

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
    coords = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
    vals = np.ascontiguousarray(pts[:, 3], dtype=np.float32)
    if pts_actor is None:
        pts_actor = Points(coords, r=4)
        # stash arrays on the actor so Python keeps them alive
        pts_actor._coords_ref = coords
        pts_actor._vals_ref = vals
        pts_actor.pointdata["val"] = vals
        pts_actor.cmap('jet', "val")
        # initialize one ray per camera
        for i in range(len(cam_pos)):
            if i < len(coords):
                ray_pts = np.ascontiguousarray([cam_pos[i], coords[i]], dtype=np.float32)
            else:
                ray_pts = np.ascontiguousarray([cam_pos[i], cam_pos[i]], dtype=np.float32)
            line = Line(ray_pts, c='black', lw=1)
            line._pts_ref = ray_pts
            ray_actors.append(line)
        plot = show([pts_actor, cam_actor, *ray_actors], axes=axes_opts, bg='white',
                    interactive=False, title='Voxel Hits')
    else:
        pts_actor.points = coords        
        pts_actor._coords_ref = coords
        pts_actor.pointdata["val"] = vals
        pts_actor._vals_ref = vals
        pts_actor.cmap('jet', "val")
        for i, line in enumerate(ray_actors):
            if i < len(coords):
                ray_pts = np.ascontiguousarray([cam_pos[i], coords[i]], dtype=np.float32)
            else:
                ray_pts = np.ascontiguousarray([cam_pos[i], cam_pos[i]], dtype=np.float32)
            line.points = ray_pts
            line._pts_ref = ray_pts
        plot.show(None, resetcam=False)
    time.sleep(0.1)

if plot is not None:
    plot.close()
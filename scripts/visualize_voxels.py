from vedo import Points, show, settings
import numpy as np, os, time, sys

settings.default_font_size = 12
PATH = os.path.join(os.path.dirname(__file__), '..', 'build', 'hits.xyz')
THRESH = 0.0

last_mtime = 0
plot = None
points = None

while True:
    if not os.path.exists(PATH):
        time.sleep(1); continue
    mtime = os.path.getmtime(PATH)
    if mtime == last_mtime:
        time.sleep(0.5); continue
    last_mtime = mtime
    pts = np.loadtxt(PATH)
    if pts.size == 0:
        time.sleep(0.5); continue
    coords = pts[:, :3]
    vals = pts[:, 3]
    if points is None:
        points = Points(coords, r=4).cmap("jet", vals)
        plot = show(points, axes=1, bg="white", interactive=False, title="Voxel Hits")
    else:
        points.points(coords)
        points.cmap("jet", vals)
        plot.show(None, resetcam=False)
    time.sleep(0.5)
import pyvista as pv
import numpy as np

# 1. Define the positions for the airplane and the camera
# ---------------------------------------------------------
# Position of the airplane in 3D space (x, y, z)

distance = 1000

start_pos = np.array([12 * distance, 5 * distance, 6 * distance])
end_pos = np.array([12 * distance, -5 * distance, 6 * distance])

airplane_position = start_pos

# Position of the camera on the ground
# We'll set z=1.7 to simulate eye-level height
camera_position = np.array([0, 0, 200])

# The point the camera is looking at (the airplane's position)
viewCenter = (start_pos + end_pos) / 2

# The "up" direction for the camera. (0, 0, 1) is standard for a ground view.
view_up = (0, 0, 1)


# 2. Load and position the airplane mesh
# ---------------------------------------------------------
# Load the default airplane mesh from pyvista's examples
airplane = pv.examples.load_airplane()

# Move the airplane to its defined position in the scene
# We add the position vector to every point in the mesh
airplane.points += airplane_position

# 3. Set up the scene and render the image
# ---------------------------------------------------------
# Initialize the plotter. off_screen=True prevents an interactive window.

video_duration = 5  # Number of frames for the animation
framerate = 30  # Frames per second
num_frames = video_duration * framerate

plane_path = np.linspace(start_pos, end_pos, num_frames)

plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

output_filename = r"C:\Users\roude\Desktop\coding stuff\targeting system vibecoding\camera_flyby.mp4"
plotter.open_movie(output_filename, framerate=30)

for i, pos in enumerate(plane_path):
    # Clear the previous frame
    plotter.clear()
    plotter.enable_lightkit()
    plotter.set_background("skyblue")
    # Update the airplane's position for each frame
    airplane.points += pos - airplane_position
    airplane_position = pos
    # Update the focal point to the current airplane position
    focal_point = viewCenter
    # Add the airplane mesh to the scene with a silver color
    plotter.add_mesh(airplane, color='silver')
    # Add a simple ground plane for context
    ground = pv.Plane(center=(0, -500, 0), direction=(0, 0, 1), i_size=4000, j_size=4000)
    plotter.add_mesh(ground, color='#46613a')  # A green-ish color for grass

    # Set the camera's position, focal point, and view-up direction
    plotter.camera_position = [camera_position, focal_point, view_up]
    # Add lighting to make the scene look more realistic
    light = pv.Light(position=(10, 10, 1000), light_type='scene light')
    plotter.add_light(light)
    # save frame to the video
    plotter.write_frame()
    print(f"Frame {i+1}/{num_frames} saved", end = "\r")

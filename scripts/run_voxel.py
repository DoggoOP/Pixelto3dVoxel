import subprocess, os, sys

def main():
    root = os.path.dirname(os.path.abspath(__file__))
    build = os.path.join(root, '..', 'build')
    os.makedirs(build, exist_ok=True)

    # Configure & build C++
    subprocess.check_call(['cmake', '..'], cwd=build)
    subprocess.check_call(['cmake', '--build', '.', '--config', 'Release'], cwd=build)

    # Run pipeline
    subprocess.check_call(['./pixeltovoxel', '../metadata.json'], cwd=build)

    # Launch visualiser (non‑blocking)
    subprocess.Popen([sys.executable, os.path.join(root, 'visualize_voxels.py')])

if __name__ == '__main__':
    main()

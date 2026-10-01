"""Shared constants, directory layout and small I/O helpers used by all pipeline stages."""

from pathlib import Path

import h5py
import numpy as np

# Table geometry (ITTF regulation sizes)
TABLE_LENGTH = 2.74  # m
TABLE_WIDTH = 1.525  # m
NET_HEIGHT = 0.1525  # m

# Frame rate of the test videos and of the simulated trajectories
FRAME_RATE = 30  # frames per second
SIM_SUBSTEPS = 3  # physics steps per video frame

# Directory layout. Inputs live in data/, everything the pipeline generates in output/, one folder per stage.
ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT_DIR / "data" / "videos"
OUTPUT_DIR = ROOT_DIR / "output"
SCENE_LIST_DIR = OUTPUT_DIR / "1_scenes"
POSE_DIR = OUTPUT_DIR / "2_poses"
POSE_DEBUG_DIR = POSE_DIR / "debug"
SIM_DATA_DIR = OUTPUT_DIR / "3_trajectories"
RECONSTRUCTION_DIR = OUTPUT_DIR / "4_reconstructions"
EVALUATION_DIR = OUTPUT_DIR / "5_evaluation"
TRAJECTORY_PLOT_DIR = EVALUATION_DIR / "trajectories"


def relative(path):
    """Path relative to the repository root, for readable console output."""
    try:
        return Path(path).resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def clear_dir(directory):
    """Create `directory` if needed and delete all files inside it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for f in directory.glob("*"):
        if f.is_file():
            f.unlink()


def save_hdf5(path, attrs=None, **datasets):
    """Write the given arrays as datasets (and optional attributes) to an HDF5 file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=np.asarray(data))
        for key, value in (attrs or {}).items():
            f.attrs[key] = value


def load_hdf5(path):
    """Read all datasets and attributes of an HDF5 file into a dict (attributes under the key "attrs")."""
    with h5py.File(path, "r") as f:
        data = {name: f[name][()] for name in f.keys()}
        data["attrs"] = dict(f.attrs)
    return data

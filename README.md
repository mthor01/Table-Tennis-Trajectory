# Table Tennis 3D Trajectory Reconstruction

Reconstruct a table-tennis ball trajectory in **3D from monocular video** by combining camera-pose estimation, a physics simulation, and evolutionary optimization.

> **Project status:** Research prototype. The current reconstruction pipeline uses simulated 3D trajectories projected into real camera poses rather than detecting the ball directly from video.

## How it works

```text
Video
  ↓
Scene / camera-cut detection
  ↓
Camera pose estimation from the table
  ↓
Physics-based trajectory simulation
  ↓
3D trajectory fitting with Differential Evolution
  ↓
Evaluation against simulated ground truth
```

The table dimensions provide a known 3D reference for estimating the camera pose. A physics simulator then models the ball's motion, including gravity, drag, spin and table bounces. Differential Evolution searches for the initial ball state whose projected 2D trajectory best matches the observations.

## Setup

The project was originally developed with **Python 3.9**.

Create and activate a virtual environment, then install the main dependencies:

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install numpy opencv-python matplotlib h5py tqdm scenedetect seaborn
```

`evaluation.py` also imports GTK through `PyGObject`. Depending on your operating system, GTK/PyGObject may need to be installed separately through the system package manager.

## Usage

Run all commands from the repository root.

### 1. Add a video

Place the input video in:

```text
single_test_vid/
```

For example:

```text
single_test_vid/test1.mp4
```

### 2. Run the pipeline

The scripts have a required execution order:

```bash
python detect_cuts.py
python pose_estimation.py
python simulation_data_gen.py
python 3d_projection_evo.py
python evaluation.py
```

Each stage writes the data needed by the next one:

| Script | Purpose | Output |
| --- | --- | --- |
| `detect_cuts.py` | Detect camera cuts/scenes | `scene_lists/` |
| `pose_estimation.py` | Estimate camera pose from table geometry | `pose_estimates/` |
| `simulation_data_gen.py` | Generate synthetic ball trajectories | `simulated_ball_data/` |
| `3d_projection_evo.py` | Recover 3D trajectories using physics + Differential Evolution | `3d_ball_positions/` |
| `evaluation.py` | Compare reconstructed and ground-truth trajectories | Evaluation plots |

## Repository structure

```text
├── single_test_vid/          # Input videos
├── scene_lists/              # Detected camera scenes
├── pose_estimates/           # Per-frame camera poses
├── simulated_ball_data/      # Synthetic trajectories and event frames
├── 3d_ball_positions/        # Reconstructed trajectories
│
├── detect_cuts.py
├── pose_estimation.py
├── simulation_data_gen.py
├── 3d_projection_evo.py
├── evaluation.py
├── classes.py                # Camera, physics simulation and optimizer classes
└── functions.py              # Shared utilities
```

## Notes

- `3d_projection_evo.py` is computationally expensive. Population size and generation count strongly affect runtime and reconstruction quality.
- Some scripts clear their output directories before generating new results, so save results you want to keep.
- Camera intrinsics are currently approximated rather than obtained from a full camera calibration.
- Real-video ball detection/tracking is not implemented yet; the current pipeline generates the 2D ball observations from simulated trajectories.

## Goal

The project explores whether **monocular 3D ball reconstruction can be constrained by physical dynamics** and how reconstruction accuracy changes with camera viewpoint.

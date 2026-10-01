# Table Tennis 3D Trajectory Reconstruction

Reconstruct a table-tennis ball trajectory in **3D from monocular video** by combining camera-pose estimation, a physics simulation, and evolutionary optimization.

> **Project status:** Research prototype from a bachelor's thesis. The reconstruction is evaluated **semi-synthetically**: camera poses are estimated from real match footage, but the ball observations are simulated trajectories projected into those cameras. Detecting the ball in real video is not implemented.

## How it works

```text
data/videos/*.mp4
  ↓  pipeline.detect_cuts             camera cuts → scenes                                     output/1_scenes/
  ↓  pipeline.estimate_poses          camera pose per scene, table as calibration object      output/2_poses/
  ↓  pipeline.simulate_trajectories   ground-truth ball flights from a physics simulation     output/3_trajectories/
  ↓  pipeline.reconstruct             2D observations → 3D (physics + Differential Evolution)  output/4_reconstructions/
  ↓  pipeline.evaluate / visualize    3D error vs camera viewpoint, trajectory images          output/5_evaluation/
```

**Pose estimation.** In one frame per scene, the table is segmented by color, its edges are found with a Hough transform, and the four corners are matched to the known table dimensions with `solvePnP`. A pose is accepted only if the reprojected table matches the detected corners. Scenes where detection fails get the pose of a similar scene, or are marked invalid.

**Reconstruction.** A single camera cannot tell how far away the ball is. The physics model fills that gap: only certain start states produce a trajectory that matches the observed 2D track. For each segment between two racket hits, Differential Evolution searches the ball's start state:

- the distance along the camera ray through the first observation, limited to a box around the table
- the start velocity
- optionally the spin

The cost is the mean pixel distance between the projected simulation and the observations. The simulation models gravity, buoyancy, drag, the Magnus force and a friction-based table bounce.

Coordinates: *world* = cm with the origin in the table center, *sim* = m with the origin in a table corner. In both, x runs along the table, y along the net and z points up.

## Repository structure

```text
├── core/                          reusable building blocks (no scripts)
│   ├── config.py                  constants, directory layout, HDF5 helpers
│   ├── geometry.py                camera model and coordinate conversions
│   ├── physics.py                 vectorised ball flight and bounce simulation
│   └── optimizer.py               Differential Evolution and the per-segment fit
├── pipeline/                      the stages, run as  python -m pipeline.<stage>
│   ├── detect_cuts.py             1
│   ├── estimate_poses.py          2
│   ├── simulate_trajectories.py   3
│   ├── reconstruct.py             4
│   ├── evaluate.py                5  error plots
│   └── visualize.py               5  trajectory images
├── data/
│   └── videos/                    input videos (test1.mp4)
├── output/                        everything the pipeline generates (not version-controlled)
│   ├── 1_scenes/                  <video>.csv
│   ├── 2_poses/                   <video>.hdf5, debug/<video>/*.jpg
│   ├── 3_trajectories/            simulated ball flights
│   ├── 4_reconstructions/         <video>__<trajectory>.hdf5
│   └── 5_evaluation/              error plots, binned_errors.csv, trajectories/*.png
├── tests/                         unit tests (pytest)
├── run_pipeline.sh                runs all stages in order
├── Dockerfile
└── requirements.txt
```

## Quick start with Docker (recommended)

You only need [Docker](https://docs.docker.com/get-docker/).

```bash
docker build -t table-tennis-trajectory .

# run the tests
docker run --rm table-tennis-trajectory python -m pytest

# quick end-to-end run (about a minute); results are written to output/ in the mounted repository
docker run --rm -v "$(pwd):/app" table-tennis-trajectory ./run_pipeline.sh --quick
```

On Windows PowerShell, use `-v "${PWD}:/app"`. The `-v` mount lets you see `output/` on your machine. Without it, the results stay inside the container.

The full run fits every segment of every trajectory. It takes much longer and uses all CPU cores:

```bash
docker run --rm -v "$(pwd):/app" table-tennis-trajectory ./run_pipeline.sh --pixel-noise 2
```

## Local setup (without Docker)

Python 3.9 or newer:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest                  # unit tests, a few seconds
./run_pipeline.sh --quick         # or run the stages below one by one
```

## Running the stages yourself

Run the stages from the repository root, in this order. Each stage deletes its previous output.

| Command | Output | Useful options |
| --- | --- | --- |
| `python -m pipeline.detect_cuts` | `output/1_scenes/` | `--threshold` |
| `python -m pipeline.estimate_poses` | `output/2_poses/` | `--debug` saves annotated frames to `output/2_poses/debug/` |
| `python -m pipeline.simulate_trajectories` | `output/3_trajectories/` | edit the grid of speeds/angles at the top of the file |
| `python -m pipeline.reconstruct` | `output/4_reconstructions/` | `--trajectories N`, `--max-segments N`, `--generations`, `--population`, `--pixel-noise PX`, `--no-spin`, `--detect-hits`, `--workers` |
| `python -m pipeline.evaluate` | `output/5_evaluation/` | |
| `python -m pipeline.visualize` | `output/5_evaluation/trajectories/` | pass result files to plot only those |

Each stage has a `--help` flag.

### What to look at

- **`output/2_poses/debug/<video>/`**: one frame per scene with the detected corners (red), the reprojected table (green) and the world axes. It quickly shows whether pose estimation worked.
- **Console output of `pipeline.reconstruct`**: the mean 3D error (cm) and reprojection error (px) for each trajectory.
- **`output/5_evaluation/`**: the 3D error grouped by camera elevation, by horizontal angle to the net, and by angle to the flight direction, plus a heatmap.
- **`output/5_evaluation/trajectories/`**: one image per result. Each shows the table, the simulated ground-truth trajectory (dashed) and the reconstruction of every fitted segment (orange, one line per segment), in a 3D view, a top view and a side view. The simulated trajectory repeats in every segment, but each segment is seen by a different camera, so the spread of the orange lines shows how much the reconstruction depends on the viewpoint.

The trajectories are **simulated**; only the camera poses come from the video. The images do not show the rally in the footage. To reconstruct the real ball, a ball detector (e.g. TrackNet) would have to provide the 2D observations.

### Things to try

- `--pixel-noise 2` simulates an imperfect ball detector. Without noise, the observations are exact and almost every fit converges to ~0 cm error.
- `--no-spin` versus the default (spin is fitted): spin adds three unknowns.
- `--detect-hits` splits the 2D track at racket hits detected in the image instead of at the known trajectory boundaries. This is experimental: under perspective, the vertical motion of a bounce can look like a direction change and cause false hits.
- Add your own videos to `data/videos/`. The pose estimation expects a blue or green table near the image center.

## Limitations

- Ball detection and tracking in real video is not implemented, so the 2D observations are synthetic.
- Camera intrinsics are approximated (square pixels, focal length = image width) instead of calibrated. Lens distortion is ignored.
- The included test video only shows very similar broadcast viewpoints. To meaningfully relate error to camera angle, you need videos with more varied angles.
- The bounce model and its coefficients come from the literature and were not validated against measured trajectories. The net is not modelled.

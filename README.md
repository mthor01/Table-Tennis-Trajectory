# Table Tennis 3D Trajectory Reconstruction

Reconstruct a table-tennis ball trajectory in **3D from monocular video** by combining camera-pose estimation, a physics simulation, and evolutionary optimization.

**Thesis:** [*3D reconstruction of table tennis trajectories from single camera footage*](Bachelors_Thesis_Git.pdf) (PDF). Bachelor's thesis by Mika Thormann, University of Tübingen, Chair of Cognitive Systems (supervisor: Prof. Dr. Andreas Zell), 2023.

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

## What a run does

The experiment asks: **how accurately can a ball trajectory be reconstructed in 3D from a single camera, and how does the accuracy depend on the camera's viewpoint?** There is no ball detector, so the pipeline combines real camera poses with simulated ball flights. That way the true 3D position of the ball is known in every frame. The numbers below are for the included `test1.mp4` and the default settings.

1. **Scenes.** `detect_cuts` splits the video at camera cuts: 28 scenes in 1809 frames. The camera is assumed not to move within a scene.
2. **One camera pose per scene.** `estimate_poses` picks the frame of each scene that shows the most table, detects the four table corners in it and computes the camera pose. Every frame of the scene gets this pose. If detection fails, the scene borrows the pose of a scene whose corners lie at the same image positions (same camera setting). If there is none, it gets the pose of the nearest scene and is marked invalid. For `test1.mp4` this gives 22 distinct poses, 16 of them valid, covering 1407 of the 1809 frames. Only the poses are stored. The detected corners appear only in the `--debug` images.
3. **Simulated shots.** `simulate_trajectories` simulates single shots, not rallies. Each starts at the end line 50 cm above the table, with 5 m/s horizontal speed and no spin, at one of 9 horizontal angles (−40° to 40°) and 5 start positions across the table. A shot ends when the ball leaves the table area or after 3 bounces. Shots shorter than 8 frames are dropped, which leaves 33 of 45 (9–30 frames each, with 1 or 2 bounces).
4. **An artificial rally across the video.** For every (video, trajectory) pair, `reconstruct` repeats the shot back to back over the whole length of the video, so the ball jumps back to its start after every pass. Each frame of the ball is projected into that frame's camera, which gives the 2D observations (optionally noisy, `--pixel-noise`). The jumps stand in for racket hits: every pass is one **segment**, and each segment is fitted on its own from its 2D observations only. Segments with any invalid frame are skipped. For `test1.mp4` there are 3220 segments over all 33 shots, and 2333 of them get fitted in the full run.
5. **Evaluation.** Because the true position is known, `evaluate` computes the 3D error in every frame. It groups the error by the viewpoint of that frame's camera: elevation above the table, horizontal angle to the net, and angle to the flight direction. `visualize` draws every fitted segment over the true shot.

### What the results show, and what they don't

The results show how well physics and Differential Evolution recover the depth that a single camera cannot see, and how this changes with viewpoint, pixel noise and fitted spin. They are not an end-to-end accuracy for real footage:

- **The camera is treated as exactly known.** The same estimated poses are used to create the observations and to fit them, so errors from pose estimation (imprecise corners, approximated intrinsics) are not part of the 3D error.
- **Few viewpoints.** Every viewpoint comes from the video's poses. That is 16 for `test1.mp4`, all of them similar broadcast angles.
- **Some segments are seen from two cameras.** A segment that crosses a camera cut is observed from two viewpoints during one flight (255 of the 2333 segments for `test1.mp4`). This resembles a stereo setup and probably makes these fits easier. The error of each frame is attributed to that frame's camera.
- **Observations are not limited to the image.** The ball is projected and used even when it would be outside the visible frame.
- **The shots are simple.** They have one speed, no spin, and all start from the same end line. Real rally shots are likely to be harder to reconstruct.

### Quick run vs. full run

|  | `./run_pipeline.sh --quick` | `./run_pipeline.sh` |
| --- | --- | --- |
| Shots | first 6 | all 33 |
| Segments per (video, shot) | at most 5 | all with a valid pose (~70 on average) |
| DE generations / population | 150 / 60 | 300 / 100 |
| Segments fitted for `test1.mp4` | 30 | 2333 |
| Run time | about a minute | much longer, uses all CPU cores |

The quick run only checks that everything works. Its segments all come from the first 5 seconds of the video. For `test1.mp4` that is the first scene, so every segment is seen from the same camera and the error-vs-viewpoint plots say nothing yet.

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
├── Bachelors_Thesis_Git.pdf       the bachelor's thesis
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

The full run fits every segment of every trajectory (see [Quick run vs. full run](#quick-run-vs-full-run)). It takes much longer and uses all CPU cores:

```bash
docker run --rm -v "$(pwd):/app" table-tennis-trajectory ./run_pipeline.sh
```

Any further arguments go to `pipeline.reconstruct`, for example `--pixel-noise 2` for noisy observations or `--workers 4` to leave some cores free.

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
- **`output/5_evaluation/trajectories/`**: one image per result. Each shows the table, the simulated ground-truth trajectory (dashed) and the reconstruction of every fitted segment (orange, one line per segment), in a 3D view, a top view and a side view. The simulated trajectory repeats in every segment, but each segment falls on a different part of the video and is seen by the camera of that scene, so the spread of the orange lines shows how much the reconstruction depends on the viewpoint.

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

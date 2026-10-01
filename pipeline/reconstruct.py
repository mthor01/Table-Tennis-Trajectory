"""
Stage 4: reconstruct 3D ball trajectories from 2D observations.

Semi-synthetic setup: every simulated trajectory is repeated over the whole length of every video and projected
into the estimated camera pose of each frame. These 2D positions (optionally with Gaussian pixel noise, see --pixel-noise) are the observations. The 2D track is split into
segments at racket hits (by default at the known repetition boundaries, see --detect-hits). Segments that contain
frames without a valid camera pose are skipped. For each remaining segment, Differential Evolution searches the start state of the ball (distance
along the camera ray through the first observation, velocity and spin) whose simulated trajectory reprojects best
onto the observations.

Output: output/4_reconstructions/<video>__<trajectory>.hdf5 with per-frame datasets "frames", "estimated" and
"ground_truth" (world coordinates in cm), "segment" and "reprojection_error". Previous results are deleted first.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from core.config import POSE_DIR, RECONSTRUCTION_DIR, SIM_DATA_DIR, clear_dir, load_hdf5, save_hdf5
from core.geometry import Camera, sim_to_world
from core.optimizer import fit_segment

MIN_SEGMENT_LENGTH = 3  # frames
MIN_HIT_GAP = 5  # frames between two detected racket hits
MIN_HIT_SPEED_CHANGE = 2  # px per frame


def load_cameras(pose_path):
    return [Camera.from_row(row) for row in load_hdf5(pose_path)["poses"]]


def observe(trajectory_sim, cameras):
    """Repeat a simulated trajectory over all frames and project it: returns (ground truth world positions, 2D)."""
    ground_truth = sim_to_world(np.resize(trajectory_sim, (len(cameras), 3)))
    observed_2d = np.array([cam.project(p[None])[0] for cam, p in zip(cameras, ground_truth)])
    return ground_truth, observed_2d


def detect_racket_hits(observed_2d, cameras):
    """
    Frames at which a new segment starts: the image-space velocity along the table's long axis changes its sign.
    Velocities across a camera cut are ignored, as they compare positions seen by different cameras.
    Experimental: vertical motion during a bounce can also flip the sign under perspective, which causes false hits.
    """
    hits = []
    previous_speed = None
    for i in range(1, len(observed_2d)):
        cam = cameras[i]
        if not np.array_equal(cam.rvec, cameras[i - 1].rvec):  # camera cut
            previous_speed = None
            continue
        axis = cam.project(np.array([[100.0, 0, 0]]))[0] - cam.project(np.zeros((1, 3)))[0]
        speed = (observed_2d[i] - observed_2d[i - 1]) @ axis / np.linalg.norm(axis)
        if not np.isfinite(speed):  # ball or table axis behind the camera
            previous_speed = None
            continue
        if previous_speed is not None and speed * previous_speed < 0 \
                and abs(speed - previous_speed) > MIN_HIT_SPEED_CHANGE \
                and (not hits or i - hits[-1] >= MIN_HIT_GAP):
            hits.append(i)
        previous_speed = speed
    return hits


def segments_from_hits(hits, num_frames):
    bounds = [0, *hits, num_frames]
    return [(start, end) for start, end in zip(bounds[:-1], bounds[1:]) if end - start >= MIN_SEGMENT_LENGTH]


def reconstruct(pose_path, trajectory_path, generations, population_size, restarts, max_segments, fit_spin,
                detect_hits, pixel_noise, seed):
    """Fit all segments of one (video, trajectory) pair and store the result. Returns a short summary."""
    start_time = time.time()
    rng = np.random.default_rng(seed)
    cameras = load_cameras(pose_path)
    trajectory = load_hdf5(trajectory_path)
    ground_truth, observed_2d = observe(trajectory["positions"], cameras)
    observed_2d = observed_2d + rng.normal(0, pixel_noise, observed_2d.shape)

    if detect_hits:
        hits = detect_racket_hits(observed_2d, cameras)
    else:  # the trajectory repeats, so a new segment starts at every repetition
        hits = list(range(len(trajectory["positions"]), len(cameras), len(trajectory["positions"])))
    all_segments = segments_from_hits(hits, len(cameras))
    segments = [(s, e) for s, e in all_segments if all(cam.valid for cam in cameras[s:e])][:max_segments]

    frames, estimated, segment_ids, reprojection_errors = [], [], [], []
    for segment_id, (start, end) in enumerate(segments):
        fit = fit_segment(observed_2d[start:end], cameras[start:end], generations, population_size, fit_spin,
                          restarts=restarts, rng=rng)
        frames.extend(range(start, end))
        estimated.append(fit.positions)
        segment_ids.extend([segment_id] * (end - start))
        reprojection_errors.extend([fit.reprojection_error] * (end - start))

    frames = np.array(frames, dtype=int)
    estimated = np.concatenate(estimated) if estimated else np.empty((0, 3))
    out_path = Path(RECONSTRUCTION_DIR, f"{pose_path.stem}__{trajectory_path.stem}.hdf5")
    save_hdf5(out_path, frames=frames, estimated=estimated, ground_truth=ground_truth[frames],
              segment=np.array(segment_ids, dtype=int), reprojection_error=np.array(reprojection_errors),
              attrs={"video": pose_path.stem, "trajectory": trajectory_path.stem, "pixel_noise": pixel_noise,
                     **trajectory["attrs"]})

    error = np.linalg.norm(estimated - ground_truth[frames], axis=1).mean() if len(frames) else float("nan")
    return (f"{out_path.name}: {len(segments)}/{len(all_segments)} segments fitted, mean 3D error {error:.1f} cm, "
            f"mean reprojection error {np.mean(reprojection_errors):.2f} px, {time.time() - start_time:.0f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations", type=int, default=300, help="DE generations per segment (default 300)")
    parser.add_argument("--population", type=int, default=100, help="DE population size (default 100)")
    parser.add_argument("--restarts", type=int, default=3,
                        help="maximum DE runs per segment while the fit is worse than 3 px (default 3)")
    parser.add_argument("--max-segments", type=int, default=None, help="fit at most this many segments per video")
    parser.add_argument("--trajectories", type=int, default=None, help="use only the first N simulated trajectories")
    parser.add_argument("--no-spin", action="store_true", help="do not fit spin (only position and velocity)")
    parser.add_argument("--detect-hits", action="store_true",
                        help="experimental: split segments at racket hits detected in 2D instead of at the known "
                             "trajectory boundaries")
    parser.add_argument("--pixel-noise", type=float, default=0.0,
                        help="standard deviation (px) of Gaussian noise added to the 2D observations (default 0)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel processes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pose_files = sorted(POSE_DIR.glob("*.hdf5"))
    trajectory_files = sorted(SIM_DATA_DIR.glob("*.hdf5"))[:args.trajectories]
    if not pose_files or not trajectory_files:
        raise SystemExit("Run python -m pipeline.estimate_poses and python -m pipeline.simulate_trajectories first")

    clear_dir(RECONSTRUCTION_DIR)
    jobs = [(p, t) for p in pose_files for t in trajectory_files]
    print(f"Reconstructing {len(jobs)} (video, trajectory) pairs with {args.workers} workers")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(reconstruct, pose_path, trajectory_path, args.generations, args.population,
                               args.restarts, args.max_segments, not args.no_spin, args.detect_hits, args.pixel_noise,
                               [args.seed, i])
                   for i, (pose_path, trajectory_path) in enumerate(jobs)]
        for future in tqdm(as_completed(futures), total=len(futures)):
            tqdm.write(future.result())


if __name__ == "__main__":
    main()

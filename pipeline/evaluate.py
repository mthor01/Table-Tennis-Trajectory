"""
Stage 5: compare reconstructed and ground-truth trajectories.

The per-frame 3D error (cm) is related to the camera viewpoint of that frame:
  * elevation: angle between the camera's viewing direction and the table plane
  * net angle: horizontal angle between the viewing direction and the net line (0 = looking along the net,
    90 = looking along the table, as from behind a player)
  * trajectory angle: horizontal angle between the viewing direction and the ball's direction of flight
Plots and a CSV with the binned values are written to output/5_evaluation/, a summary is printed to the console.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from core.config import EVALUATION_DIR, POSE_DIR, RECONSTRUCTION_DIR, load_hdf5, relative  # noqa: E402
from core.geometry import Camera, angle_between  # noqa: E402

BIN_WIDTH = 10  # degrees
NUM_BINS = 90 // BIN_WIDTH

# chart styling
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def fold(angle):
    """Map an angle between two lines (0..180 degrees) to 0..90 degrees."""
    return 180 - angle if angle > 90 else angle


def camera_angles(cam, trajectory_angle):
    view = cam.viewing_direction
    horizontal = np.array([view[0], view[1]])
    elevation = angle_between(view, [view[0], view[1], 0])
    net_angle = fold(angle_between(horizontal, [0, 1]))
    flight = [np.cos(np.radians(trajectory_angle)), np.sin(np.radians(trajectory_angle))]
    return elevation, net_angle, fold(angle_between(horizontal, flight))


def to_bin(angle):
    return min(int(angle // BIN_WIDTH), NUM_BINS - 1)


def collect(result_files):
    """Per-frame errors and camera angles of all reconstruction results."""
    rows = []
    cameras_cache = {}
    for path in result_files:
        result = load_hdf5(path)
        attrs = result["attrs"]
        video = attrs["video"]
        if video not in cameras_cache:
            cameras_cache[video] = [Camera.from_row(r) for r in load_hdf5(Path(POSE_DIR, video + ".hdf5"))["poses"]]
        cameras = cameras_cache[video]

        errors = np.linalg.norm(result["estimated"] - result["ground_truth"], axis=1)
        for frame, error in zip(result["frames"], errors):
            cam = cameras[frame]
            if cam.valid:
                rows.append((video, attrs["trajectory"], error, *camera_angles(cam, attrs["angle"])))
    return rows


def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)


def bar_chart(errors, angles, title, xlabel, path):
    """Mean error per angle bin; the number of frames per bin is written above each bar."""
    bins = np.array([to_bin(a) for a in angles])
    means = [errors[bins == b].mean() if np.any(bins == b) else np.nan for b in range(NUM_BINS)]
    counts = [int(np.sum(bins == b)) for b in range(NUM_BINS)]
    centers = np.arange(NUM_BINS) * BIN_WIDTH + BIN_WIDTH / 2

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    heights = np.nan_to_num(means)
    ax.bar(centers, heights, width=BIN_WIDTH * 0.8, color=SERIES, zorder=2)
    for x, h, n in zip(centers, heights, counts):
        if n:
            ax.annotate(f"n={n}", (x, h), xytext=(0, 4), textcoords="offset points", ha="center",
                        color=INK_MUTED, fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_xticks(np.arange(0, 91, BIN_WIDTH))
    ax.set_xlim(0, 90)
    style_axes(ax, title, xlabel, "Mean 3D error (cm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return list(zip(range(0, 90, BIN_WIDTH), means, counts))


def heatmap(errors, elevations, net_angles, path):
    grid_sum = np.zeros((NUM_BINS, NUM_BINS))
    grid_count = np.zeros((NUM_BINS, NUM_BINS))
    for error, elevation, net_angle in zip(errors, elevations, net_angles):
        grid_sum[to_bin(elevation), to_bin(net_angle)] += error
        grid_count[to_bin(elevation), to_bin(net_angle)] += 1
    with np.errstate(invalid="ignore"):
        grid = np.where(grid_count > 0, grid_sum / grid_count, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL)
    cmap.set_bad(SURFACE)
    image = ax.imshow(grid, origin="lower", cmap=cmap, extent=(0, 90, 0, 90), vmin=0)
    for i in range(NUM_BINS):
        for j in range(NUM_BINS):
            if grid_count[i, j]:
                value = grid[i, j]
                dark_cell = value > 0.6 * np.nanmax(grid) if np.nanmax(grid) > 0 else False
                ax.text(j * BIN_WIDTH + BIN_WIDTH / 2, i * BIN_WIDTH + BIN_WIDTH / 2, f"{value:.1f}",
                        ha="center", va="center", fontsize=7, color="#ffffff" if dark_cell else INK)
    ax.set_xticks(np.arange(0, 91, BIN_WIDTH))
    ax.set_yticks(np.arange(0, 91, BIN_WIDTH))
    style_axes(ax, "Mean 3D error (cm) by camera viewpoint",
               "Horizontal angle between viewing direction and net (°)",
               "Elevation of viewing direction above table plane (°)")
    ax.spines["bottom"].set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=INK_MUTED, length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    result_files = sorted(RECONSTRUCTION_DIR.glob("*.hdf5"))
    if not result_files:
        raise SystemExit("No reconstructions found, run python -m pipeline.reconstruct first")
    rows = collect(result_files)
    if not rows:
        raise SystemExit("No reconstructed frames with a valid camera pose")

    videos, trajectories, errors, elevations, net_angles, trajectory_angles = zip(*rows)
    errors = np.array(errors)

    print(f"{len(errors)} frames from {len(result_files)} reconstructions")
    print(f"3D error: mean {errors.mean():.2f} cm, median {np.median(errors):.2f} cm, "
          f"90th percentile {np.percentile(errors, 90):.2f} cm, max {errors.max():.2f} cm")
    print(f"frames with an error below 5 cm: {np.mean(errors < 5) * 100:.1f} %")
    for video in sorted(set(videos)):
        mask = np.array(videos) == video
        print(f"  {video}: mean {errors[mask].mean():.2f} cm over {mask.sum()} frames")

    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    charts = {
        "elevation": bar_chart(errors, elevations, "Error vs camera elevation",
                               "Angle between viewing direction and table plane (°)",
                               EVALUATION_DIR / "error_vs_elevation.png"),
        "net_angle": bar_chart(errors, net_angles, "Error vs horizontal camera angle",
                               "Horizontal angle between viewing direction and net (°)",
                               EVALUATION_DIR / "error_vs_net_angle.png"),
        "trajectory_angle": bar_chart(errors, trajectory_angles, "Error vs angle to the ball's flight direction",
                                      "Horizontal angle between viewing direction and trajectory (°)",
                                      EVALUATION_DIR / "error_vs_trajectory_angle.png"),
    }
    heatmap(errors, elevations, net_angles, EVALUATION_DIR / "error_heatmap.png")

    with open(EVALUATION_DIR / "binned_errors.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["angle_type", "bin_start_deg", "bin_end_deg", "mean_error_cm", "frames"])
        for name, binned in charts.items():
            for start, mean, count in binned:
                writer.writerow([name, start, start + BIN_WIDTH, "" if np.isnan(mean) else f"{mean:.3f}", count])
    print(f"Plots and binned_errors.csv written to {relative(EVALUATION_DIR)}/")


if __name__ == "__main__":
    main()

"""
Plot reconstruction results from output/4_reconstructions/ as images.

For every result file (or the files given as arguments) output/5_evaluation/trajectories/<name>.png shows the simulated ground-truth
trajectory and the reconstruction of every fitted segment on the table, in a 3D view, a top view and a side view.

The trajectories are simulated (see pipeline/reconstruct.py); only the camera poses used for the reconstruction come
from the video.

Examples:
  python -m pipeline.visualize                                     images for all results
  python -m pipeline.visualize output/4_reconstructions/test1__y=0.02_speed=5.0_angle=0.0.hdf5
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from core.config import (NET_HEIGHT, RECONSTRUCTION_DIR, TABLE_LENGTH, TABLE_WIDTH,  # noqa: E402
                         TRAJECTORY_PLOT_DIR, load_hdf5, relative)

# colors: the reconstruction is the highlighted series, the ground truth a neutral reference
ESTIMATE = "#eb6834"
GROUND_TRUTH = "#0b0b0b"
TABLE = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

HALF_LENGTH = TABLE_LENGTH * 50  # cm
HALF_WIDTH = TABLE_WIDTH * 50  # cm
NET_TOP = NET_HEIGHT * 100  # cm
TABLE_OUTLINE = np.array([(HALF_LENGTH, -HALF_WIDTH, 0), (HALF_LENGTH, HALF_WIDTH, 0), (-HALF_LENGTH, HALF_WIDTH, 0),
                          (-HALF_LENGTH, -HALF_WIDTH, 0), (HALF_LENGTH, -HALF_WIDTH, 0)])
NET_OUTLINE = np.array([(0, -HALF_WIDTH, 0), (0, -HALF_WIDTH, NET_TOP), (0, HALF_WIDTH, NET_TOP),
                        (0, HALF_WIDTH, 0)])


def segment_ranges(segments):
    """(segment id, slice) for each run of equal segment ids."""
    starts = np.flatnonzero(np.diff(segments, prepend=-1))
    ends = np.append(starts[1:], len(segments))
    return [(segments[s], slice(s, e)) for s, e in zip(starts, ends)]


def unique_paths(paths):
    """Drop paths that are (almost) identical to an earlier one, e.g. the repeated ground truth."""
    kept = []
    for path in paths:
        if not any(len(k) == len(path) and np.allclose(k, path, atol=1e-6) for k in kept):
            kept.append(path)
    return kept


def style_2d(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_3d(ax, truths, estimates):
    ax.set_facecolor(SURFACE)
    xx, yy = np.meshgrid([-HALF_LENGTH, HALF_LENGTH], [-HALF_WIDTH, HALF_WIDTH])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color=TABLE, alpha=0.15, shade=False)
    ax.plot(*TABLE_OUTLINE.T, color=TABLE, linewidth=1.5)
    ax.plot(*NET_OUTLINE.T, color=INK_MUTED, linewidth=1.5)

    for path in estimates:
        ax.plot(*path.T, color=ESTIMATE, linewidth=1.2, alpha=0.6)
    for path in truths:
        ax.plot(*path.T, color=GROUND_TRUTH, linewidth=1.8, linestyle="--")

    points = np.concatenate([TABLE_OUTLINE, *truths, *estimates])
    lower = np.minimum(points.min(axis=0), [-HALF_LENGTH, -HALF_WIDTH, 0]) - [20, 20, 0]
    upper = np.maximum(points.max(axis=0), [HALF_LENGTH, HALF_WIDTH, NET_TOP]) + [20, 20, 10]
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    ax.set_box_aspect(upper - lower, zoom=1.25)
    for artist in ax.lines + ax.collections:  # with zoom > 1, matplotlib would clip them at the unzoomed box
        artist.set_clip_on(False)
    ax.view_init(elev=22, azim=-60)
    ax.set_title("3D view", loc="left", color=INK, fontsize=11)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis._axinfo["grid"]["color"] = GRID
    ax.set_xlabel("x (cm)", color=INK_SECONDARY)
    ax.set_ylabel("y (cm)", color=INK_SECONDARY)
    ax.set_zlabel("z (cm)", color=INK_SECONDARY, labelpad=-2)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def plot_top(ax, truths, estimates):
    ax.add_patch(Rectangle((-HALF_LENGTH, -HALF_WIDTH), 2 * HALF_LENGTH, 2 * HALF_WIDTH, facecolor=TABLE,
                           alpha=0.15, edgecolor=TABLE, linewidth=1.5))
    ax.plot([0, 0], [-HALF_WIDTH, HALF_WIDTH], color=INK_MUTED, linewidth=1.5)
    for path in estimates:
        ax.plot(path[:, 0], path[:, 1], color=ESTIMATE, linewidth=1.2, alpha=0.6)
    for path in truths:
        ax.plot(path[:, 0], path[:, 1], color=GROUND_TRUTH, linewidth=1.8, linestyle="--")
    ax.set_xlim(-HALF_LENGTH - 30, HALF_LENGTH + 30)
    ax.set_ylim(-HALF_WIDTH - 20, HALF_WIDTH + 20)
    ax.set_aspect("equal")
    style_2d(ax, "Top view", "x along the table (cm)", "y along the net (cm)")


def plot_side(ax, truths, estimates):
    ax.plot([-HALF_LENGTH, HALF_LENGTH], [0, 0], color=TABLE, linewidth=3, solid_capstyle="butt")
    ax.plot([0, 0], [0, NET_TOP], color=INK_MUTED, linewidth=2)
    for path in estimates:
        ax.plot(path[:, 0], path[:, 2], color=ESTIMATE, linewidth=1.2, alpha=0.6)
    for path in truths:
        ax.plot(path[:, 0], path[:, 2], color=GROUND_TRUTH, linewidth=1.8, linestyle="--")
    top = max(NET_TOP, *(p[:, 2].max() for p in truths + estimates))
    bottom = min(0, *(p[:, 2].min() for p in truths + estimates))
    ax.set_xlim(-HALF_LENGTH - 30, HALF_LENGTH + 30)
    ax.set_ylim(bottom - 10, top + 15)
    ax.set_aspect("equal")
    style_2d(ax, "Side view", "x along the table (cm)", "z height (cm)")


def plot_result(result, path):
    attrs = result["attrs"]
    estimated, truth = result["estimated"], result["ground_truth"]
    errors = np.linalg.norm(estimated - truth, axis=1)
    ranges = segment_ranges(result["segment"])
    estimates = [estimated[s] for _, s in ranges]
    truths = unique_paths([truth[s] for _, s in ranges])

    fig = plt.figure(figsize=(14, 10.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], top=0.89, bottom=0.06, left=0.06, right=0.97,
                            hspace=0.12, wspace=0.18)
    plot_3d(fig.add_subplot(grid[0, :], projection="3d"), truths, estimates)
    plot_top(fig.add_subplot(grid[1, 0]), truths, estimates)
    plot_side(fig.add_subplot(grid[1, 1]), truths, estimates)

    noise = f", {attrs['pixel_noise']:g} px observation noise" if attrs.get("pixel_noise") else ""
    fig.suptitle(f"{attrs['trajectory']}  ·  camera poses from {attrs['video']}", x=0.03, y=0.98, ha="left",
                 color=INK, fontsize=14)
    fig.text(0.03, 0.935,
             f"{len(ranges)} reconstructed segments{noise}  ·  mean 3D error {errors.mean():.1f} cm, "
             f"median {np.median(errors):.1f} cm, max {errors.max():.1f} cm",
             color=INK_SECONDARY, fontsize=10)
    legend = [Line2D([], [], color=GROUND_TRUTH, linewidth=1.8, linestyle="--", label="Simulated ground truth"),
              Line2D([], [], color=ESTIMATE, linewidth=1.5, label="Reconstruction (one line per segment)"),
              Line2D([], [], color=TABLE, linewidth=3, label="Table"),
              Line2D([], [], color=INK_MUTED, linewidth=2, label="Net")]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.98, 0.985), ncol=2, frameon=False,
               labelcolor=INK, fontsize=10)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="*", type=Path, help="result files (default: all in output/4_reconstructions/)")
    args = parser.parse_args()

    results = args.results or sorted(RECONSTRUCTION_DIR.glob("*.hdf5"))
    if not results:
        raise SystemExit("No reconstructions found, run python -m pipeline.reconstruct first")
    TRAJECTORY_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for path in results:
        result = load_hdf5(path)
        if len(result["frames"]) == 0:
            print(f"{path.name}: no fitted frames, skipped")
            continue
        image_path = TRAJECTORY_PLOT_DIR / f"{path.stem}.png"
        plot_result(result, image_path)
        print(f"wrote {relative(image_path)}")


if __name__ == "__main__":
    main()

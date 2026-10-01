"""Tests for the stage scripts that do not need the test video."""

import cv2 as cv
import numpy as np

from core.geometry import Camera
from pipeline.detect_cuts import read_scene_list, write_scene_list
from pipeline.estimate_poses import TABLE_CORNERS_3D, estimate_scene_pose, table_color_range
from pipeline.reconstruct import detect_racket_hits, observe, segments_from_hits
from pipeline.simulate_trajectories import simulate_trajectory
from pipeline.visualize import segment_ranges
from test_geometry import POSE_ROW


def render_table(cam):
    """A red floor with a blue table seen from `cam` (BGR image)."""
    img = np.full((cam.img_height, cam.img_width, 3), (40, 40, 200), np.uint8)
    corners = cam.project(TABLE_CORNERS_3D[[0, 3, 1, 2]])  # outline order
    cv.fillPoly(img, [corners.astype(np.int32)], (200, 90, 40))
    return img


def test_pose_is_recovered_from_rendered_table():
    cam = Camera.from_row(POSE_ROW)
    img = render_table(cam)
    pose = estimate_scene_pose(img, table_color_range(img), cam.intrinsics)

    assert pose.valid
    estimated = Camera(pose.rvec, pose.tvec, True, cam.img_height, cam.img_width)
    # same camera position up to a few cm (the table is ~7 m away) or the 180 degree symmetric solution
    candidates = [cam.position, cam.position * [-1, -1, 1]]
    assert min(np.linalg.norm(estimated.position - c) for c in candidates) < 10
    assert estimated.position[2] > 0


def test_scene_list_round_trip(tmp_path):
    scenes = [(0, 330), (330, 360), (360, 420)]
    write_scene_list(tmp_path / "scenes.csv", scenes)
    assert read_scene_list(tmp_path / "scenes.csv") == scenes


def test_simulated_trajectory_stays_over_table_and_bounces():
    positions, bounces = simulate_trajectory([0.02, 0.76, 0.5], [5.0, 0.0, -2.0])
    assert len(positions) > 10
    assert len(bounces) >= 1
    assert np.all(positions[:, 2] > 0)


def test_segments_from_hits():
    assert segments_from_hits([10, 12, 30], 40) == [(0, 10), (12, 30), (30, 40)]  # (10, 12) is too short


def test_racket_hits_found_at_trajectory_repetitions():
    cam = Camera.from_row(POSE_ROW)
    trajectory, _ = simulate_trajectory([0.02, 0.76, 0.5], [5.0, 0.0, -2.0])
    _, observed = observe(trajectory, [cam] * (3 * len(trajectory)))
    hits = detect_racket_hits(observed, [cam] * len(observed))
    assert hits == [len(trajectory), 2 * len(trajectory)]


def test_visualize_segment_ranges():
    ranges = segment_ranges(np.array([0, 0, 0, 1, 1, 3]))
    assert [(seg, (s.start, s.stop)) for seg, s in ranges] == [(0, (0, 3)), (1, (3, 5)), (3, (5, 6))]

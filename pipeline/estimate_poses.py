"""
Stage 2: estimate the camera pose for every frame of every video.

The table is used as calibration object: in one representative frame per scene the table is segmented by its
color, its edges are found with a Hough transform, and the four corners are matched to the known table
dimensions with solvePnP. Every frame of a scene gets the pose of its scene.

Output per video: output/2_poses/<video>.hdf5 with a dataset "poses" of shape (num_frames, 9),
one row [tvec(3), rvec(3), valid, img_height, img_width] per frame (world coordinates in cm, table center origin).
Previously stored pose estimates are deleted first. Use --debug to save annotated frames to output/2_poses/debug/.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from core.config import POSE_DEBUG_DIR, POSE_DIR, SCENE_LIST_DIR, VIDEO_DIR, clear_dir, relative, save_hdf5
from pipeline.detect_cuts import read_scene_list
from core.geometry import Camera, approx_intrinsics

# 3D table corners in world coordinates (cm). Corners 0/1 and 2/3 are diagonally opposite.
TABLE_CORNERS_3D = np.array([(137, -76.25, 0),
                             (-137, 76.25, 0),
                             (-137, -76.25, 0),
                             (137, 76.25, 0)], dtype=float)

NO_DISTORTION = np.zeros((4, 1))
MIN_SATURATION = 30
SHORT_SCENE_LENGTH = 40  # frames; in shorter scenes simply the middle frame is used
HOUGH_SENSITIVITIES = range(12, 27, 2)  # higher sensitivity => lower vote threshold => more lines
MAX_LINES = 5


@dataclass
class Line:
    pos1: np.ndarray
    pos2: np.ndarray


@dataclass
class IntersecPoint:
    pos: np.ndarray
    line1_index: int
    line2_index: int


@dataclass
class ScenePose:
    rvec: np.ndarray = None
    tvec: np.ndarray = None
    corners: np.ndarray = None  # (4, 2) detected image corners, None if no corners were found
    valid: bool = False


# --- table segmentation ---------------------------------------------------------------------------------------------

def table_color_range(img):
    """HSV bounds for a blue or green table, depending on which color dominates the image center."""
    h, w = img.shape[:2]
    hsv = cv.cvtColor(img[h // 4: h * 3 // 4, w // 4: w * 3 // 4], cv.COLOR_BGR2HSV)
    hue = hsv[..., 0][hsv[..., 1] > MIN_SATURATION]
    blue = np.count_nonzero((hue > 90) & (hue < 135))
    green = np.count_nonzero((hue > 20) & (hue < 75))

    if blue >= green:
        return np.array([90, 30, 90]), np.array([135, 255, 255])
    return np.array([40, 30, 90]), np.array([75, 255, 255])


def color_mask(img, color_range):
    return cv.inRange(cv.cvtColor(img, cv.COLOR_BGR2HSV), *color_range)


def table_contours(contours, img_width, img_height, min_second_size=0.25):
    """
    Pick the contours of the table: the largest contour whose bounding box covers the image center, plus the second
    largest one if it is at least `min_second_size` times as large (the net often splits the table into two halves).
    If no contour covers the center, the one whose bounding box is closest to the center is used.
    """
    center = np.array([img_width / 2, img_height / 2])
    covering = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if x <= center[0] <= x + w and y <= center[1] <= y + h:
            covering.append((cv.contourArea(contour), contour))

    if not covering:
        if not contours:
            return []

        def center_distance(contour):
            x, y, w, h = cv.boundingRect(contour)
            return norm(np.clip(center, [x, y], [x + w, y + h]) - center)

        return [min(contours, key=center_distance)]

    covering.sort(key=lambda item: item[0], reverse=True)
    selected = [covering[0][1]]
    if len(covering) > 1 and covering[1][0] > covering[0][0] * min_second_size:
        selected.append(covering[1][1])
    return selected


# --- lines and corners ----------------------------------------------------------------------------------------------

def hough_lines(edge_img, sensitivity):
    """Hough lines clipped to the image borders; lines whose end points are close to an existing line are dropped."""
    img_height, img_width = edge_img.shape[:2]
    found = cv.HoughLines(edge_img, 1, np.pi / 360, int(img_height / sensitivity))
    lines = []
    if found is None:
        return lines

    for rho, theta in found[:, 0]:
        origin = np.array([np.cos(theta), np.sin(theta)]) * rho
        direction = np.array([-np.sin(theta), np.cos(theta)])
        p1 = tuple(int(v) for v in origin - 10000 * direction)
        p2 = tuple(int(v) for v in origin + 10000 * direction)
        inside, p1, p2 = cv.clipLine((0, 0, img_width, img_height), p1, p2)
        if not inside:
            continue
        p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)

        duplicate = False
        for line in lines:
            a, b = (line.pos1, line.pos2) if norm(p1 - line.pos1) <= norm(p1 - line.pos2) else (line.pos2, line.pos1)
            if norm(p1 - a) < img_width / 20 and norm(p2 - b) < img_width / 20:
                duplicate = True
                break
        if not duplicate:
            lines.append(Line(p1, p2))
    return lines


def line_intersection(line1, line2, img_width, img_height):
    """Intersection point of two (infinite) lines, or None if they are parallel or it lies outside the image."""
    d1 = line1.pos2 - line1.pos1
    d2 = line2.pos2 - line2.pos1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-9:
        return None
    diff = line2.pos1 - line1.pos1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    p = line1.pos1 + t * d1
    if 0 <= p[0] <= img_width and 0 <= p[1] <= img_height:
        return p
    return None


def intersections(lines, img_width, img_height):
    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            p = line_intersection(lines[i], lines[j], img_width, img_height)
            if p is not None:
                points.append(IntersecPoint(p, i, j))
    return points


def extreme_corners(points):
    """The left-, right-, top- and bottom-most of the given points (each point used at most once)."""
    chosen = []
    for key in (lambda p: p.pos[0], lambda p: -p.pos[0], lambda p: p.pos[1], lambda p: -p.pos[1]):
        candidates = [p for p in points if all(p is not c for c in chosen)]
        chosen.append(min(candidates, key=key))
    return chosen


def order_corners(corners):
    """
    Reorder corners such that 0/1 and 2/3 are diagonally opposite, i.e. corner 1 shares no line with corner 0.
    Returns None if the corners do not form a quadrilateral of four distinct lines.
    """
    if len({c.line1_index for c in corners} | {c.line2_index for c in corners}) != 4:
        return None
    first_lines = {corners[0].line1_index, corners[0].line2_index}
    opposite = [c for c in corners[1:] if not ({c.line1_index, c.line2_index} & first_lines)]
    if len(opposite) != 1:
        return None
    others = [c for c in corners[1:] if c is not opposite[0]]
    return [corners[0], opposite[0], *others]


def min_pairwise_distance(points):
    return min(norm(points[i] - points[j]) for i in range(len(points)) for j in range(i + 1, len(points)))


# --- pose -----------------------------------------------------------------------------------------------------------

def solve_table_pose(corners_2d, camera_matrix):
    """
    solvePnP for the four table corners. As the order of the second diagonal is unknown, both assignments are tried
    and the one with the smaller maximum reprojection error is kept.
    Returns (rvec, tvec, max reprojection error in px). The rotation is flipped if needed so that z points up.
    """
    best = None
    for order in ([0, 1, 2, 3], [0, 1, 3, 2]):
        pts = np.ascontiguousarray(corners_2d[order], dtype=float)
        ok, rvec, tvec = cv.solvePnP(TABLE_CORNERS_3D, pts, camera_matrix, NO_DISTORTION, flags=cv.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        projected, _ = cv.projectPoints(TABLE_CORNERS_3D, rvec, tvec, camera_matrix, NO_DISTORTION)
        error = norm(projected[:, 0] - pts, axis=1).max()
        if best is None or error < best[2]:
            best = (rvec, tvec, error)
    if best is None:
        return None

    rvec, tvec, error = best
    rotation, _ = cv.Rodrigues(rvec)
    # the table is symmetric: rotate by 180 degrees around its x axis if the z axis points downwards in the image
    if rotation[1, 2] > 0:
        rvec, _ = cv.Rodrigues(rotation @ np.diag([1.0, -1.0, -1.0]))
    return rvec.reshape(3), tvec.reshape(3), error


def estimate_scene_pose(img, color_range, camera_matrix, debug_img=None):
    """Estimate the camera pose from a single image showing the table."""
    img_height, img_width = img.shape[:2]

    mask = cv.morphologyEx(color_mask(img, color_range), cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contour_img = cv.drawContours(np.zeros((img_height, img_width), np.uint8),
                                  table_contours(contours, img_width, img_height), -1, 255, 1)

    result = ScenePose()
    for sensitivity in HOUGH_SENSITIVITIES:
        lines = hough_lines(contour_img, sensitivity)
        points = intersections(lines, img_width, img_height)
        if len(points) >= 4:
            corners = order_corners(extreme_corners(points))
            if corners is not None:
                corners_2d = np.array([c.pos for c in corners])
                pose = solve_table_pose(corners_2d, camera_matrix)
                if pose is not None:
                    rvec, tvec, error = pose
                    result = ScenePose(rvec, tvec, corners_2d, False)
                    # accept the pose if the reprojected table matches the detected corners
                    if error < img_height / 15 and min_pairwise_distance(corners_2d) > img_height / 10:
                        result.valid = True
                        break
        if len(lines) > MAX_LINES:  # more lines will not lead to a better solution
            break

    if debug_img is not None:
        for line in lines:
            cv.line(debug_img, tuple(line.pos1.astype(int)), tuple(line.pos2.astype(int)), (255, 0, 0), 2, cv.LINE_AA)
        if result.rvec is not None:
            draw_table(debug_img, result, camera_matrix)
    return result


# --- drawing --------------------------------------------------------------------------------------------------------

def draw_table(img, pose, camera_matrix):
    """Detected corners (red), reprojected table outline (green) and the world axes (x red, y green, z blue)."""
    for p in pose.corners:
        cv.circle(img, tuple(p.astype(int)), 6, (0, 0, 255), -1)
    outline = np.array([(137, -76.25, 0), (137, 76.25, 0), (-137, 76.25, 0), (-137, -76.25, 0)], dtype=float)
    projected, _ = cv.projectPoints(outline, pose.rvec, pose.tvec, camera_matrix, NO_DISTORTION)
    cv.polylines(img, [projected[:, 0].astype(np.int32)], True, (0, 255, 0), 2)

    axes = np.array([(0, 0, 0), (100, 0, 0), (0, 50, 0), (0, 0, 50)], dtype=float)
    projected, _ = cv.projectPoints(axes, pose.rvec, pose.tvec, camera_matrix, NO_DISTORTION)
    o, x, y, z = (tuple(p.astype(int)) for p in projected[:, 0])
    for end, color in ((x, (0, 0, 255)), (y, (0, 255, 0)), (z, (255, 0, 0))):
        cv.line(img, o, end, color, 3)
    text = "valid" if pose.valid else "rejected"
    cv.putText(img, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if pose.valid else (0, 0, 255), 2)


# --- whole video ----------------------------------------------------------------------------------------------------

def representative_frames(video_path, scenes, color_range):
    """
    For every scene pick one frame with little occlusion of the table: the frame (away from the cuts) that contains
    the most table-colored pixels. In short scenes the middle frame is used.
    Returns a list of (frame_index, image).
    """
    video = cv.VideoCapture(str(video_path))
    chosen = []
    frame_index = 0
    for start, end in scenes:
        best_img, best_index, best_count = None, start, -1
        short = end - start < SHORT_SCENE_LENGTH
        while frame_index < end:
            ok, img = video.read()
            if not ok:
                break
            if short:
                if frame_index == (start + end) // 2 or best_img is None:
                    best_img, best_index = img, frame_index
            elif start + 30 < frame_index < end - 10:
                count = cv.countNonZero(color_mask(img, color_range))
                if count > best_count:
                    best_img, best_index, best_count = img, frame_index, count
            elif best_img is None:
                best_img, best_index = img, frame_index
            frame_index += 1
        chosen.append((best_index, best_img))
    video.release()
    return chosen


def fill_failed_scenes(scene_poses, img_height):
    """
    Scenes without a reliable pose get the pose of a reliable scene whose detected corners are mostly at the same
    positions (same camera setting). Scenes that still have no pose at all get the pose of the closest scene that has
    one, but stay marked as invalid.
    """
    for i, pose in enumerate(scene_poses):
        if pose.valid or pose.corners is None:
            continue
        best_match, best_similar = None, 1
        for other in scene_poses:
            if other.valid:
                similar = int(np.sum(norm(pose.corners - other.corners, axis=1) < img_height / 40))
                if similar > best_similar:
                    best_match, best_similar = other, similar
        if best_match is not None:
            scene_poses[i] = best_match

    with_pose = [i for i, p in enumerate(scene_poses) if p.rvec is not None]
    if not with_pose:
        raise RuntimeError("The table could not be found in any scene")
    for i, pose in enumerate(scene_poses):
        if pose.rvec is None:
            nearest = scene_poses[min(with_pose, key=lambda j: abs(i - j))]
            scene_poses[i] = ScenePose(nearest.rvec, nearest.tvec, nearest.corners, False)
    return scene_poses


def estimate_poses(video_path, scenes, debug_dir=None):
    """Returns an array (num_frames, 9) with one pose row per frame."""
    video = cv.VideoCapture(str(video_path))
    ok, first_img = video.read()
    video.release()
    if not ok:
        raise RuntimeError(f"Could not read {video_path}")
    img_height, img_width = first_img.shape[:2]
    camera_matrix = approx_intrinsics(img_width, img_height)
    color_range = table_color_range(first_img)

    scene_poses = []
    frames = representative_frames(video_path, scenes, color_range)
    for scene_index, (frame_index, img) in enumerate(tqdm(frames, desc="scenes", leave=False)):
        debug_img = img.copy() if debug_dir else None
        scene_poses.append(estimate_scene_pose(img, color_range, camera_matrix, debug_img))
        if debug_dir:
            cv.imwrite(str(Path(debug_dir, f"scene{scene_index:03d}_frame{frame_index:05d}.jpg")), debug_img)

    num_detected = sum(p.valid for p in scene_poses)
    scene_poses = fill_failed_scenes(scene_poses, img_height)
    num_valid = sum(p.valid for p in scene_poses)
    print(f"{Path(video_path).name}: table found in {num_detected}/{len(scenes)} scenes, "
          f"{num_valid}/{len(scenes)} valid after filling in similar scenes")

    rows = []
    for (start, end), pose in zip(scenes, scene_poses):
        row = Camera(pose.rvec, pose.tvec, pose.valid, img_height, img_width).to_row()
        rows.extend([row] * (end - start))
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true", help=f"save annotated scene frames to {relative(POSE_DEBUG_DIR)}/")
    args = parser.parse_args()

    clear_dir(POSE_DIR)
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    for video_path in tqdm(videos, desc="videos"):
        scene_list = Path(SCENE_LIST_DIR, video_path.stem + ".csv")
        if not scene_list.exists():
            raise SystemExit(f"Missing {relative(scene_list)}, run python -m pipeline.detect_cuts first")
        debug_dir = None
        if args.debug:
            debug_dir = Path(POSE_DEBUG_DIR, video_path.stem)
            clear_dir(debug_dir)

        poses = estimate_poses(video_path, read_scene_list(scene_list), debug_dir)
        save_hdf5(Path(POSE_DIR, video_path.stem + ".hdf5"), poses=poses)


if __name__ == "__main__":
    main()

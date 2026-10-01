"""
Stage 1: detect camera cuts in every video of the video directory.

For each video a CSV file with one row per scene (start_frame, end_frame; 0-based, end exclusive)
is written to the scene list directory. Previously stored scene lists are deleted first.
"""

import argparse
import csv
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from core.config import SCENE_LIST_DIR, VIDEO_DIR, clear_dir, relative

THRESHOLD = 8.0
MIN_SCENE_LENGTH = 0


def detect_scenes(video_path, threshold=THRESHOLD):
    """Return a list of (start_frame, end_frame) tuples covering the whole video."""
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=MIN_SCENE_LENGTH))
    scene_manager.detect_scenes(video)
    scenes = [(start.get_frames(), end.get_frames()) for start, end in scene_manager.get_scene_list()]
    if not scenes:  # no cut at all: the whole video is a single scene
        scenes = [(0, video.duration.get_frames())]
    return scenes


def write_scene_list(path, scenes):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_frame", "end_frame"])
        writer.writerows(scenes)


def read_scene_list(path):
    with open(path, newline="") as f:
        return [(int(row["start_frame"]), int(row["end_frame"])) for row in csv.DictReader(f)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="ContentDetector threshold")
    args = parser.parse_args()

    clear_dir(SCENE_LIST_DIR)
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 videos found in {relative(VIDEO_DIR)}/")

    for video_path in videos:
        scenes = detect_scenes(video_path, args.threshold)
        write_scene_list(Path(SCENE_LIST_DIR, video_path.stem + ".csv"), scenes)
        print(f"{video_path.name}: {len(scenes)} scenes")


if __name__ == "__main__":
    main()

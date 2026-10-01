"""
Stage 3: simulate ground-truth ball trajectories.

Every combination of start speed, horizontal angle and lateral start position is simulated, starting at the end line
of the table. A simulation stops once the ball leaves the table area or after MAX_BOUNCES bounces.
Output: output/3_trajectories/<name>.hdf5 with the datasets "positions" (num_frames, 3; sim coordinates in m)
and "bounce_frames", and the start parameters as attributes. Previously generated data is deleted first.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np

from core.config import (FRAME_RATE, SIM_DATA_DIR, SIM_SUBSTEPS, TABLE_LENGTH, TABLE_WIDTH, clear_dir, relative,
                         save_hdf5)
from core.physics import BALL_RADIUS, BallSimulation

TEST_SPEEDS = [5.0]  # m/s, horizontal speed
TEST_ANGLES = [-40, -30, -20, -10, 0, 10, 20, 30, 40]  # degrees, horizontal angle to the table's long axis
START_POSITIONS_Y = np.linspace(0.02, TABLE_WIDTH - 0.02, 5)  # m
START_POSITION_X = 0.02  # m
START_POSITION_Z = 0.5  # m above the table
START_SPEED_Z = -2.0  # m/s
START_SPIN = [0.0, 0.0, 0.0]  # rad/s

MAX_FRAMES = 500
MAX_BOUNCES = 3
MIN_FRAMES = 8  # shorter trajectories are not stored (too few observations for 7 unknowns)


def simulate_trajectory(start_position, start_velocity, start_spin=START_SPIN):
    """Positions at every video frame and the frames at which the ball bounced."""
    sim = BallSimulation([start_position], [start_velocity], [start_spin])
    dt = 1 / (FRAME_RATE * SIM_SUBSTEPS)
    positions, bounce_frames = [], []

    for frame in range(MAX_FRAMES):
        x, y, _ = sim.positions[0]
        if not (0 < x < TABLE_LENGTH and 0 < y < TABLE_WIDTH):
            break
        positions.append(sim.positions[0].copy())
        for _ in range(SIM_SUBSTEPS):
            sim.step(dt)
            if sim.bounced[0]:
                bounce_frames.append(frame)
        if len(bounce_frames) >= MAX_BOUNCES:
            break

    return np.array(positions), np.array(bounce_frames, dtype=int)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    clear_dir(SIM_DATA_DIR)
    num_stored = 0
    for speed, angle, y0 in itertools.product(TEST_SPEEDS, TEST_ANGLES, START_POSITIONS_Y):
        rad = np.radians(angle)
        velocity = [speed * np.cos(rad), speed * np.sin(rad), START_SPEED_Z]
        start = [START_POSITION_X, y0, START_POSITION_Z]
        positions, bounce_frames = simulate_trajectory(start, velocity)
        if len(positions) < MIN_FRAMES:
            continue

        name = f"y={y0:.2f}_speed={speed:.1f}_angle={angle:.1f}"
        save_hdf5(Path(SIM_DATA_DIR, name + ".hdf5"), positions=positions, bounce_frames=bounce_frames,
                  attrs={"start_y": y0, "speed": speed, "angle": angle, "ball_radius": BALL_RADIUS})
        num_stored += 1

    print(f"Stored {num_stored} trajectories in {relative(SIM_DATA_DIR)}/")


if __name__ == "__main__":
    main()

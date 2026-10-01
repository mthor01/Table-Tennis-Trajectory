"""
Table tennis ball flight simulation, vectorised over many balls at once.

Forces: gravity, buoyancy, air drag and the Magnus force caused by spin.
Bounces on the table use a discrete momentum-theorem model with a velocity dependent coefficient of friction
that distinguishes between a sliding and a rolling contact.
All quantities are in sim coordinates (m, origin in a table corner, z up).
"""

import numpy as np

from core.config import TABLE_LENGTH, TABLE_WIDTH

GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.2  # kg/m^3
DRAG_COEFFICIENT = 0.4
MAGNUS_COEFFICIENT = 0.069
BALL_MASS = 0.0027  # kg
BALL_RADIUS = 0.02  # m
COR = 0.9  # coefficient of restitution of the table
BOUNCE_COOLDOWN = 3  # simulation steps after a bounce during which no further bounce is applied

_CROSS_SECTION = np.pi * BALL_RADIUS ** 2
_VOLUME = 4 / 3 * np.pi * BALL_RADIUS ** 3


class BallSimulation:
    """Simulates N balls in parallel. positions, velocities and spins are arrays of shape (N, 3)."""

    def __init__(self, positions, velocities, spins=None):
        self.positions = np.array(positions, dtype=float).reshape(-1, 3)
        self.velocities = np.array(velocities, dtype=float).reshape(-1, 3)
        if spins is None:
            spins = np.zeros_like(self.positions)
        self.spins = np.array(spins, dtype=float).reshape(-1, 3)  # angular velocities in rad/s
        self.steps_since_bounce = np.full(len(self.positions), BOUNCE_COOLDOWN + 1)
        self.bounced = np.zeros(len(self.positions), dtype=bool)

    def accelerations(self):
        speed = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        drag = -0.5 * AIR_DENSITY * _CROSS_SECTION * DRAG_COEFFICIENT * speed * self.velocities
        magnus = (4 / 3) * MAGNUS_COEFFICIENT * AIR_DENSITY * np.pi * BALL_RADIUS ** 3 \
            * np.cross(self.spins, self.velocities)
        buoyancy = np.array([0.0, 0.0, AIR_DENSITY * _VOLUME * GRAVITY])
        return np.array([0.0, 0.0, -GRAVITY]) + (drag + magnus + buoyancy) / BALL_MASS

    def step(self, dt):
        """Advance all balls by dt seconds (semi-implicit Euler) and apply table bounces."""
        self.velocities = self.velocities + self.accelerations() * dt
        self.positions = self.positions + self.velocities * dt
        self.steps_since_bounce += 1
        self._table_bounce()
        return self.positions

    def _table_bounce(self):
        x, y, z = self.positions.T
        over_table = (x > 0) & (x < TABLE_LENGTH) & (y > 0) & (y < TABLE_WIDTH)
        contact = over_table & (z < BALL_RADIUS) & (self.velocities[:, 2] < 0)
        self.bounced = contact & (self.steps_since_bounce > BOUNCE_COOLDOWN)
        if not self.bounced.any():
            return

        idx = self.bounced
        self.velocities[idx], self.spins[idx] = bounce(self.velocities[idx], self.spins[idx])
        self.positions[idx, 2] = BALL_RADIUS
        self.steps_since_bounce[idx] = 0


def bounce(velocities, spins):
    """
    Velocities and spins (N, 3) after a bounce on the table.
    The tangential part follows a discrete momentum-theorem model: depending on the friction impulse the ball either
    keeps sliding over the whole contact or starts rolling.
    """
    v_in = velocities.copy()
    w_in = spins.copy()
    r = BALL_RADIUS
    vx, vy, vz = v_in.T
    wx, wy, _ = w_in.T

    # velocity of the contact point (bottom of the ball) relative to the table
    contact_vx = vx - wy * r
    contact_vy = vy + wx * r
    contact_speed = np.hypot(contact_vx, contact_vy)

    # empirical coefficient of friction, fitted on contact speeds in km/h
    cof = 0.0017 * contact_speed * 3.6 + 0.1635
    normal_impulse = (1 + COR) * np.abs(vz)

    with np.errstate(divide="ignore", invalid="ignore"):
        slide_ratio = cof * normal_impulse / contact_speed
    rolling = ~(slide_ratio < 0.4)  # also True for contact_speed == 0
    safe_speed = np.where(contact_speed > 0, contact_speed, 1.0)

    v_out = v_in.copy()
    w_out = w_in.copy()
    v_out[:, 2] = COR * np.abs(vz)

    # rolling: the contact point comes to rest during the bounce
    v_out[rolling, 0] = 0.6 * vx[rolling] + 0.4 * wy[rolling] * r
    v_out[rolling, 1] = 0.6 * vy[rolling] - 0.4 * wx[rolling] * r
    w_out[rolling, 0] = 0.4 * wx[rolling] - 0.6 * vy[rolling] / r
    w_out[rolling, 1] = 0.4 * wy[rolling] + 0.6 * vx[rolling] / r

    # sliding: friction acts against the contact point velocity over the whole contact
    # (the contact point velocity shrinks by a factor (1 - 2.5 * friction), which is 0 exactly at the rolling limit)
    s = ~rolling
    friction = cof[s] * normal_impulse[s] / safe_speed[s]
    v_out[s, 0] = vx[s] - friction * contact_vx[s]
    v_out[s, 1] = vy[s] - friction * contact_vy[s]
    w_out[s, 0] = wx[s] - 1.5 * friction * contact_vy[s] / r
    w_out[s, 1] = wy[s] + 1.5 * friction * contact_vx[s] / r

    return v_out, w_out


def simulate(positions, velocities, spins, num_frames, frame_rate, substeps):
    """
    Simulate N balls and return their positions at every video frame as an array (num_frames, N, 3).
    Frame 0 is the start state.
    """
    sim = BallSimulation(positions, velocities, spins)
    dt = 1 / (frame_rate * substeps)
    out = np.empty((num_frames, len(sim.positions), 3))
    for frame in range(num_frames):
        out[frame] = sim.positions
        for _ in range(substeps):
            sim.step(dt)
    return out

"""
3D trajectory fitting: Differential Evolution searches the start state of the ball whose simulated
trajectory, projected into the per-frame cameras, best matches the observed 2D ball positions.
"""

from dataclasses import dataclass

import numpy as np

from core.config import FRAME_RATE, SIM_SUBSTEPS
from core.geometry import project_points, sim_to_world, world_to_sim
from core.physics import simulate

MAX_SPEED = 10.0  # m/s, search bound per velocity component
MISSING_PROJECTION_PENALTY = 1e4  # px, used when a simulated point lies behind the camera

# box (world coordinates in cm) in which the ball can be at the start of a segment:
# the table plus 1 m on every side, from the floor up to 2.5 m above the table
PLAY_VOLUME = (np.array([-237.0, -176.25, -76.0]), np.array([237.0, 176.25, 250.0]))


def ray_box_interval(origin, direction, box_min, box_max):
    """Distances (near, far) along a ray inside an axis-aligned box; far <= near if the ray misses the box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box_min - origin) / direction
        t2 = (box_max - origin) / direction
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    near = max(np.max(np.minimum(t1, t2)), 0.0)
    far = np.min(np.maximum(t1, t2))
    return near, far


class DifferentialEvolution:
    """
    Classic DE/rand/1/bin minimiser for a vectorised objective.
    `objective` gets an array (population_size, dims) and returns one cost per row.
    """

    def __init__(self, objective, lower, upper, population_size=100, F=0.5, CR=0.9, rng=None):
        self.objective = objective
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.rng = np.random.default_rng() if rng is None else rng

        dims = len(self.lower)
        self.population = self.lower + self.rng.random((population_size, dims)) * (self.upper - self.lower)
        self.costs = self.objective(self.population)

    def _mutants(self):
        n, dims = self.population.shape
        # three distinct partners per individual, none of them the individual itself
        partners = np.array([self.rng.choice(n - 1, 3, replace=False) for _ in range(n)])
        partners += partners >= np.arange(n)[:, None]
        r1, r2, r3 = partners.T
        mutants = self.population[r1] + self.F * (self.population[r2] - self.population[r3])

        # binomial crossover; every trial vector takes at least one component from its mutant
        cross = self.rng.random((n, dims)) < self.CR
        cross[np.arange(n), self.rng.integers(0, dims, n)] = True
        trials = np.where(cross, mutants, self.population)
        return np.clip(trials, self.lower, self.upper)

    def step(self):
        trials = self._mutants()
        trial_costs = self.objective(trials)
        better = trial_costs < self.costs
        self.population[better] = trials[better]
        self.costs[better] = trial_costs[better]

    def run(self, generations):
        for _ in range(generations):
            self.step()
        best = int(np.argmin(self.costs))
        return self.population[best], self.costs[best]


@dataclass
class SegmentFit:
    positions: np.ndarray  # (num_frames, 3) reconstructed world positions in cm
    projections: np.ndarray  # (num_frames, 2) reprojected pixel positions
    reprojection_error: float  # mean pixel error
    start_velocity: np.ndarray  # m/s, sim coordinates
    start_spin: np.ndarray  # rad/s, sim coordinates


class SegmentModel:
    """
    Maps a parameter vector to a trajectory for one segment between two racket hits.
    The start position is restricted to the camera ray through the first observed ball position, so only its
    distance along the ray is searched: params = [ray_fraction, vx, vy, vz(, wx, wy, wz)].
    """

    def __init__(self, observed_2d, cameras, fit_spin=True, max_spin=150.0):
        self.observed_2d = np.asarray(observed_2d, dtype=float)
        self.num_frames = len(self.observed_2d)
        self.fit_spin = fit_spin
        self.rotations = np.stack([c.rotation_matrix for c in cameras])
        self.tvecs = np.stack([c.tvec for c in cameras])
        self.intrinsics = np.stack([c.intrinsics for c in cameras])

        first_cam = cameras[0]
        self.ray_origin = first_cam.position
        self.ray_direction = first_cam.pixel_ray(self.observed_2d[0])
        # only the part of the ray inside the play volume is searched (cm along the ray)
        self.ray_near, self.ray_far = ray_box_interval(self.ray_origin, self.ray_direction, *PLAY_VOLUME)
        if self.ray_far <= self.ray_near:  # the ray misses the play volume: fall back to a generous range
            self.ray_near, self.ray_far = 0.0, 2 * np.linalg.norm(first_cam.tvec)

        lower = [0.0, -MAX_SPEED, -MAX_SPEED, -MAX_SPEED]
        upper = [1.0, MAX_SPEED, MAX_SPEED, MAX_SPEED]
        if fit_spin:
            lower += [-max_spin] * 3
            upper += [max_spin] * 3
        self.lower, self.upper = np.array(lower), np.array(upper)

    def start_states(self, params):
        params = np.atleast_2d(params)
        distance = self.ray_near + params[:, :1] * (self.ray_far - self.ray_near)
        start_world = self.ray_origin + distance * self.ray_direction
        velocities = params[:, 1:4]
        spins = params[:, 4:7] if self.fit_spin else np.zeros_like(velocities)
        return world_to_sim(start_world), velocities, spins

    def trajectories(self, params):
        """World positions (num_frames, N, 3) and their projections (num_frames, N, 2)."""
        positions_sim = simulate(*self.start_states(params), self.num_frames, FRAME_RATE, SIM_SUBSTEPS)
        positions_world = sim_to_world(positions_sim)
        projections = project_points(positions_world, self.rotations, self.tvecs, self.intrinsics)
        return positions_world, projections

    def cost(self, params):
        """Mean pixel distance between projected trajectory and observations for every parameter vector."""
        _, projections = self.trajectories(params)
        errors = np.linalg.norm(projections - self.observed_2d[:, None, :], axis=2)
        errors = np.where(np.isfinite(errors), errors, MISSING_PROJECTION_PENALTY)
        return errors.mean(axis=0)


def fit_segment(observed_2d, cameras, generations=300, population_size=100, fit_spin=True, max_spin=150.0,
                restarts=3, good_enough=3.0, rng=None):
    """
    Reconstruct the 3D trajectory of one segment from its 2D observations and per-frame cameras.
    DE is restarted up to `restarts` times while the mean reprojection error is above `good_enough` pixels,
    as single runs occasionally converge to a local minimum (e.g. a mirrored trajectory far behind the table).
    """
    rng = np.random.default_rng() if rng is None else rng
    model = SegmentModel(observed_2d, cameras, fit_spin, max_spin)
    best, cost = None, np.inf
    for _ in range(max(restarts, 1)):
        de = DifferentialEvolution(model.cost, model.lower, model.upper, population_size, rng=rng)
        candidate, candidate_cost = de.run(generations)
        if candidate_cost < cost:
            best, cost = candidate, candidate_cost
        if cost < good_enough:
            break

    positions, projections = model.trajectories(best)
    return SegmentFit(
        positions=positions[:, 0],
        projections=projections[:, 0],
        reprojection_error=float(cost),
        start_velocity=best[1:4],
        start_spin=best[4:7] if fit_spin else np.zeros(3),
    )

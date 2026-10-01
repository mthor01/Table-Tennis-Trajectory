import numpy as np

from core.config import FRAME_RATE, SIM_SUBSTEPS
from core.geometry import Camera, sim_to_world
from core.optimizer import DifferentialEvolution, fit_segment, ray_box_interval
from core.physics import simulate
from test_geometry import POSE_ROW


def test_de_minimises_sphere_function():
    de = DifferentialEvolution(lambda x: np.sum((x - 1.5) ** 2, axis=1), [-5] * 4, [5] * 4, population_size=30,
                               rng=np.random.default_rng(0))
    best, cost = de.run(200)
    np.testing.assert_allclose(best, 1.5, atol=1e-3)
    assert cost < 1e-6


def test_de_respects_bounds():
    de = DifferentialEvolution(lambda x: x[:, 0], [2.0], [3.0], population_size=10, rng=np.random.default_rng(0))
    best, _ = de.run(50)
    assert best[0] >= 2.0
    assert np.all((de.population >= 2.0) & (de.population <= 3.0))


def test_ray_box_interval():
    near, far = ray_box_interval(np.array([-10.0, 0, 0]), np.array([1.0, 0, 0]), np.array([-1.0] * 3),
                                 np.array([1.0] * 3))
    assert (near, far) == (9.0, 11.0)
    near, far = ray_box_interval(np.array([-10.0, 5, 0]), np.array([1.0, 0, 0]), np.array([-1.0] * 3),
                                 np.array([1.0] * 3))
    assert far <= near  # miss


def test_fit_segment_recovers_synthetic_trajectory():
    cam = Camera.from_row(POSE_ROW)
    num_frames = 20
    truth_sim = simulate([[0.05, 0.6, 0.45]], [[4.5, 0.8, -1.5]], [[0, 0, 0]], num_frames, FRAME_RATE, SIM_SUBSTEPS)
    truth_world = sim_to_world(truth_sim[:, 0])
    observed = cam.project(truth_world)

    fit = fit_segment(observed, [cam] * num_frames, generations=200, population_size=60, fit_spin=False,
                      rng=np.random.default_rng(1))
    assert fit.reprojection_error < 0.5
    assert np.linalg.norm(fit.positions - truth_world, axis=1).mean() < 2.0  # cm
    np.testing.assert_allclose(fit.start_velocity, [4.5, 0.8, -1.5], atol=0.1)

import numpy as np

from core.physics import BALL_RADIUS, COR, BallSimulation, bounce, simulate


def test_free_fall_without_air_is_close_to_parabola():
    # a slow ball: drag is negligible over 0.1 s
    out = simulate([[1.0, 0.7, 1.0]], [[0.0, 0.0, 0.0]], None, num_frames=4, frame_rate=30, substeps=30)
    t = 3 / 30
    assert abs(out[3, 0, 2] - (1.0 - 0.5 * 9.81 * t ** 2)) < 2e-3


def test_drag_slows_the_ball():
    sim = BallSimulation([[0.5, 0.7, 1.0]], [[10.0, 0.0, 0.0]])
    for _ in range(30):
        sim.step(1 / 90)
    assert sim.velocities[0, 0] < 10.0


def test_topspin_pushes_ball_down():
    # for a ball travelling in +x, a spin vector along +y is topspin: the Magnus force w x v points down
    no_spin = simulate([[0.2, 0.7, 0.5]], [[5.0, 0.0, 1.0]], [[0, 0, 0]], 10, 30, 3)
    topspin = simulate([[0.2, 0.7, 0.5]], [[5.0, 0.0, 1.0]], [[0, 150.0, 0]], 10, 30, 3)
    assert topspin[-1, 0, 2] < no_spin[-1, 0, 2]


def test_vertical_bounce_keeps_restitution():
    v, w = bounce(np.array([[0.0, 0.0, -3.0]]), np.zeros((1, 3)))
    np.testing.assert_allclose(v[0], [0, 0, COR * 3.0])
    np.testing.assert_allclose(w[0], [0, 0, 0])


def test_bounce_reduces_contact_point_velocity():
    for vx in (0.5, 3.0, 10.0):  # rolling and sliding regime
        v_in, w_in = np.array([[vx, 0.0, -3.0]]), np.zeros((1, 3))
        v, w = bounce(v_in, w_in)
        contact_in = v_in[0, 0] - w_in[0, 1] * BALL_RADIUS
        contact_out = v[0, 0] - w[0, 1] * BALL_RADIUS
        assert 0 <= contact_out < contact_in  # friction slows the contact point but never reverses it
        assert w[0, 1] > 0  # friction makes the ball roll forward


def test_ball_bounces_on_table_but_not_beside_it():
    sim = BallSimulation([[1.0, 0.7, 0.1], [-0.5, 0.7, 0.1]], [[0, 0, -2.0], [0, 0, -2.0]])
    bounced = np.zeros(2, dtype=bool)
    lowest = np.full(2, np.inf)
    for _ in range(30):
        sim.step(1 / 90)
        bounced |= sim.bounced
        lowest = np.minimum(lowest, sim.positions[:, 2])
    assert bounced[0] and lowest[0] >= BALL_RADIUS - 1e-9  # bounced on the table
    assert not bounced[1] and sim.positions[1, 2] < 0  # fell past the table edge

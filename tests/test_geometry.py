import numpy as np

from core.geometry import Camera, angle_between, sim_to_world, world_to_sim

# a real pose estimated from the test video: [tvec(3), rvec(3), valid, height, width]
POSE_ROW = [18.45, 6.25, 691.24, 0.9826, 1.8221, -1.4537, 1.0, 720.0, 1280.0]


def test_sim_world_round_trip():
    points = np.array([[0, 0, 0], [1.37, 0.7625, 0.5], [2.74, 1.525, 1.0]])
    np.testing.assert_allclose(world_to_sim(sim_to_world(points)), points)
    # the table center is the world origin
    np.testing.assert_allclose(sim_to_world([1.37, 0.7625, 0]), [0, 0, 0], atol=1e-12)


def test_camera_row_round_trip():
    cam = Camera.from_row(POSE_ROW)
    np.testing.assert_allclose(cam.to_row(), POSE_ROW)


def test_camera_transforms_are_inverse():
    cam = Camera.from_row(POSE_ROW)
    points = np.array([[10.0, -20.0, 5.0], [100.0, 50.0, 30.0]])
    np.testing.assert_allclose(cam.cam_to_world(cam.world_to_cam(points)), points, atol=1e-9)
    np.testing.assert_allclose(cam.world_to_cam(cam.position), [0, 0, 0], atol=1e-9)


def test_table_center_is_visible_and_camera_above_table():
    cam = Camera.from_row(POSE_ROW)
    u, v = cam.project(np.zeros((1, 3)))[0]
    assert 0 < u < cam.img_width and 0 < v < cam.img_height
    assert cam.position[2] > 0  # z points up and the camera looks down on the table


def test_pixel_ray_hits_projected_point():
    cam = Camera.from_row(POSE_ROW)
    point = np.array([50.0, -30.0, 40.0])
    pixel = cam.project(point[None])[0]
    ray = cam.pixel_ray(pixel)
    distance = np.linalg.norm(point - cam.position)
    np.testing.assert_allclose(cam.position + distance * ray, point, atol=1e-6)


def test_angle_between():
    assert angle_between([1, 0], [0, 1]) == 90
    assert abs(angle_between([1, 0, 0], [1, 1, 0]) - 45) < 1e-9

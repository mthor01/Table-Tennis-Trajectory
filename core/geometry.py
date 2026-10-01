"""
Camera model and coordinate conversions.

Two 3D coordinate systems are used throughout the project:
  * world: origin in the middle of the table surface, units in cm (used by pose estimation)
  * sim:   origin in a table corner, units in m (used by the physics simulation)
Both have the x axis along the table length, y along the table width and z pointing up.
"""

import cv2 as cv
import numpy as np

from core.config import TABLE_LENGTH, TABLE_WIDTH

# offset of the world origin expressed in sim coordinates (m)
_TABLE_CENTER_SIM = np.array([TABLE_LENGTH / 2, TABLE_WIDTH / 2, 0.0])


def sim_to_world(points):
    """Convert points (..., 3) from sim coordinates (m, corner origin) to world coordinates (cm, center origin)."""
    return (np.asarray(points, dtype=float) - _TABLE_CENTER_SIM) * 100


def world_to_sim(points):
    """Convert points (..., 3) from world coordinates (cm, center origin) to sim coordinates (m, corner origin)."""
    return np.asarray(points, dtype=float) / 100 + _TABLE_CENTER_SIM


def approx_intrinsics(img_width, img_height):
    """
    Approximate camera matrix used in place of a real calibration:
    square pixels, principal point in the image center and a focal length equal to the image width
    (roughly a 53 degree horizontal field of view).
    """
    f = float(img_width)
    return np.array([[f, 0, img_width / 2],
                     [0, f, img_height / 2],
                     [0, 0, 1]])


def angle_between(vec1, vec2):
    """Angle between two vectors in degrees."""
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


class Camera:
    """
    Camera pose of a single frame.
    rvec/tvec follow the OpenCV convention: x_cam = R @ x_world + tvec, with world coordinates in cm.
    """

    def __init__(self, rvec, tvec, valid, img_height, img_width):
        self.rvec = np.asarray(rvec, dtype=float).reshape(3)
        self.tvec = np.asarray(tvec, dtype=float).reshape(3)
        self.valid = bool(valid)
        self.img_height = int(img_height)
        self.img_width = int(img_width)
        self.intrinsics = approx_intrinsics(self.img_width, self.img_height)
        self.rotation_matrix, _ = cv.Rodrigues(self.rvec)

    # a pose row as stored in output/2_poses/*.hdf5: [tvec(3), rvec(3), valid, img_height, img_width]
    @classmethod
    def from_row(cls, row):
        return cls(row[3:6], row[0:3], row[6], row[7], row[8])

    def to_row(self):
        return np.concatenate((self.tvec, self.rvec, [float(self.valid), self.img_height, self.img_width]))

    @property
    def position(self):
        """Camera center in world coordinates (cm)."""
        return -self.rotation_matrix.T @ self.tvec

    @property
    def viewing_direction(self):
        """Unit vector of the optical axis in world coordinates."""
        return self.rotation_matrix.T @ np.array([0.0, 0.0, 1.0])

    def world_to_cam(self, points):
        return np.asarray(points, dtype=float) @ self.rotation_matrix.T + self.tvec

    def cam_to_world(self, points):
        return (np.asarray(points, dtype=float) - self.tvec) @ self.rotation_matrix

    def project(self, points_world):
        """Project world points (..., 3) to pixel coordinates (..., 2)."""
        return project_points(points_world, self.rotation_matrix, self.tvec, self.intrinsics)

    def pixel_ray(self, pixel):
        """Unit direction (world coordinates) of the ray from the camera center through a pixel."""
        direction_cam = np.linalg.inv(self.intrinsics) @ np.array([pixel[0], pixel[1], 1.0])
        direction_world = self.rotation_matrix.T @ direction_cam
        return direction_world / np.linalg.norm(direction_world)


def project_points(points_world, rotation, tvec, intrinsics):
    """
    Pinhole projection without distortion. Works on batches:
    points_world (..., N, 3) with rotation (..., 3, 3), tvec (..., 3) and intrinsics (..., 3, 3) broadcast over the
    leading dimensions. Points behind the camera get +inf coordinates.
    """
    points_world = np.asarray(points_world, dtype=float)
    cam = np.einsum("...ij,...nj->...ni", rotation, points_world) + np.asarray(tvec)[..., None, :]
    img = np.einsum("...ij,...nj->...ni", intrinsics, cam)
    z = img[..., 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.where(z > 1e-6, img[..., :2] / z, np.inf)
    return uv

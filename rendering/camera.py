from __future__ import annotations

import numpy as np

from config.enums import CameraMovement
from utils.transform import *


class Camera:
    def __init__(self, config=None):
        self.aspect_ratio = 1.0
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # Internal state vectors
        self.position = np.array(config.position, dtype=np.float32)
        self.front = np.array(config.front, dtype=np.float32)
        self.up = np.array(config.up, dtype=np.float32)
        self.right = np.array(config.right, dtype=np.float32)
        self.yaw = np.float32(config.yaw)
        self.pitch = np.float32(config.pitch)

        self.fov = np.float32(config.fov)
        self.near_plane = np.float32(config.near_plane)
        self.far_plane = np.float32(config.far_plane)
        self.move_speed = np.float32(config.move_speed)
        self.sensitivity = np.float32(config.sensitivity)

    def move(self, movement, step_scale=1.0):
        velocity = float(step_scale) * float(self.move_speed)
        if velocity == 0:
            return
        if movement is CameraMovement.FORWARD:
            displacement = self.front * velocity
        elif movement is CameraMovement.BACKWARD:
            displacement = -self.front * velocity
        elif movement is CameraMovement.LEFT:
            displacement = -self.right * velocity
        elif movement is CameraMovement.RIGHT:
            displacement = self.right * velocity
        elif movement is CameraMovement.UP:
            displacement = self.up * velocity
        elif movement is CameraMovement.DOWN:
            displacement = -self.up * velocity
        else:
            return

        self.position = self.position + displacement
        self._recalculate_basis()

    def look(self, old, new):
        offset = (new[0] - old[0], new[1] - old[1])
        yaw_offset = np.float32(offset[0]) * self.sensitivity
        pitch_offset = np.float32(offset[1]) * self.sensitivity
        if yaw_offset == 0 and pitch_offset == 0:
            return
        self.yaw += -yaw_offset
        self.pitch += pitch_offset

        # Clamp pitch to avoid flipping
        if self.pitch > 89.0:
            self.pitch = 89.0
        if self.pitch < -89.0:
            self.pitch = -89.0

        cp = np.cos(np.radians(self.pitch))
        cy = np.cos(np.radians(self.yaw))
        sp = np.sin(np.radians(self.pitch))
        sy = np.sin(np.radians(self.yaw))

        # Imagine the camera rotation as a sphere
        self.front = np.array(
            [
                cp * cy,
                sp,
                cp * sy,
            ],
            dtype=np.float32,
        )
        self._recalculate_basis()

    def pan(self, old, new):
        """Pan the camera in the right/up plane."""
        offset = (new[0] - old[0], new[1] - old[1])
        if offset[0] == 0 and offset[1] == 0:
            return

        # Pan sensitivity factor (adjust this for faster/slower panning)
        pan_speed = 0.005

        # Move along the right and up vectors
        right_offset = -offset[0] * pan_speed
        up_offset = offset[1] * pan_speed

        self.position = (
            self.position + (self.right * right_offset) + (self.up * up_offset)
        )
        self._recalculate_basis()

    def get_view_matrix(self):
        forward = self.front
        right = self.right
        up = self.up
        rotate = np.identity(4, dtype=np.float32)
        rotate[0, 0:3] = right
        rotate[1, 0:3] = up
        rotate[2, 0:3] = -forward
        translate = np.identity(4, dtype=np.float32)
        translate[0:3, 3] = -self.position
        return np.dot(rotate, translate)

    def get_projection_matrix(self):
        # fov_rad = np.radians(self.fov)
        # f = float(1.0 / np.tan(fov_rad / 2.0))
        # near = float(self.near_plane)
        # far = float(self.far_plane)

        # proj = np.zeros((4, 4), dtype=np.float32)
        # proj[0, 0] = f / float(self.aspect_ratio)
        # proj[1, 1] = f
        # proj[2, 2] = (far + near) / (near - far)
        # proj[2, 3] = (2.0 * far * near) / (near - far)
        # proj[3, 2] = -1.0
        # return proj
        return perspective(self.fov, self.aspect_ratio, self.near_plane, self.far_plane)

    def _recalculate_basis(self):
        self.front = self._safe_normalize(self.front)
        self.right = self._safe_normalize(np.cross(self.front, self.world_up))
        if np.linalg.norm(self.right) < 1e-6:
            self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.up = self._safe_normalize(np.cross(self.right, self.front))
        if np.linalg.norm(self.up) < 1e-6:
            self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    @staticmethod
    def _safe_normalize(vector: np.ndarray):
        norm = float(np.linalg.norm(vector))
        if norm < 1e-6:
            return np.zeros_like(vector)
        return vector / norm


# a trackball class based on provided quaternion functions -------------------
class Trackball:
    """Virtual trackball for 3D scene viewing. Independent of windows system."""

    def __init__(self, config=None):
        """Build a new trackball with specified view, angles in degrees"""
        self.position = config.position
        self.rotation = quaternion_from_euler(
            config.yaw, config.roll, config.pitch, config.radians
        )
        self.distance = max(config.distance, 0.001)
        self.pos2d = vec(0.0, 0.0)
        self.pan_sensitivity = config.pan_sensitivity

    def drag(self, old, new, winsize):
        """Move trackball from old to new 2d normalized windows position"""
        # Convert to normalized coordinates
        winsize_vec = vec(winsize)
        old_vec = (2 * vec(old) - winsize_vec) / winsize_vec
        new_vec = (2 * vec(new) - winsize_vec) / winsize_vec
        self.rotation = quaternion_mul(self._rotate(old_vec, new_vec), self.rotation)

    def zoom(self, delta, size):
        """Zoom trackball by a factor delta normalized by windows size"""
        # Extract scalar values from the vec arrays
        delta_vec = vec(delta)
        size_vec = vec(size)
        # Use the y component of delta (index 1) for zoom
        zoom_factor = 1 - 50 * (delta_vec[1] / size_vec[1])
        self.distance = max(0.001, self.distance * zoom_factor)

    def pan(self, old, new):
        """Pan in camera's reference by a 2d vector factor of (new - old)."""
        delta = vec(new) - vec(old)
        self.pos2d += delta * self.pan_sensitivity * self.distance

    def get_view_matrix(self):
        """View matrix transformation, including distance to target point"""
        return translate(*self.pos2d, -self.distance) @ self._matrix()

    def get_projection_matrix(self, winsize):
        """Projection matrix with z-clipping range adaptive to distance"""
        z_range = vec(0.1, 100) * self.distance  # proportion to dist
        return perspective(35, winsize[0] / winsize[1], *z_range)

    def get_camera_position(self):
        """World-space position of the virtual camera."""
        view = self.get_view_matrix()
        inv_view = np.linalg.inv(view)
        origin = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return (inv_view @ origin)[:3]

    def _matrix(self):
        """Rotational component of trackball position"""
        return quaternion_matrix(self.rotation)

    def _project3d(self, position2d, radius=0.8):
        """Project x,y on sphere OR hyperbolic sheet if away from center"""
        p2, r2 = sum(position2d * position2d), radius * radius
        zcoord = math.sqrt(r2 - p2) if 2 * p2 < r2 else r2 / (2 * math.sqrt(p2))
        return vec(*position2d, zcoord)

    def _rotate(self, old, new):
        """Rotation of axis orthogonal to old & new's 3D ball projections"""
        old, new = (normalized(self._project3d(pos)) for pos in (old, new))
        phi = 2 * math.acos(np.clip(np.dot(old, new), -1, 1))
        return quaternion_from_axis_angle(np.cross(old, new), radians=phi)

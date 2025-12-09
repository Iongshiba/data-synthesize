"""
Object placement utilities for collision detection and surface attachment
"""

import numpy as np
from typing import Optional, Tuple, List
from scipy.spatial import KDTree


def ray_triangle_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> Optional[Tuple[float, np.ndarray]]:
    """
    Möller–Trumbore ray-triangle intersection algorithm

    Returns:
        (t, intersection_point) if intersection exists, None otherwise
    """
    epsilon = 1e-6

    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(ray_direction, edge2)
    a = np.dot(edge1, h)

    if abs(a) < epsilon:
        return None  # Ray is parallel to triangle

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return None

    q = np.cross(s, edge1)
    v = f * np.dot(ray_direction, q)

    if v < 0.0 or u + v > 1.0:
        return None

    t = f * np.dot(edge2, q)

    if t > epsilon:
        intersection_point = ray_origin + ray_direction * t
        return t, intersection_point

    return None


def find_surface_point(
    mesh_vertices: np.ndarray,
    mesh_indices: np.ndarray,
    position_2d: Tuple[float, float],
    camera_pos: np.ndarray,
    camera_dir: np.ndarray,
    up_dir: np.ndarray = np.array([0, 1, 0]),
) -> Optional[np.ndarray]:
    """
    Cast a ray from camera through the 2D position and find intersection with mesh

    Args:
        mesh_vertices: Nx3 array of vertex positions
        mesh_indices: Mx3 array of triangle indices
        position_2d: (x, z) position in ground plane
        camera_pos: Camera position
        camera_dir: Camera forward direction
        up_dir: Up direction (usually Y+)

    Returns:
        3D intersection point on mesh surface, or None
    """
    # Cast ray downward from above the position
    ray_origin = np.array([position_2d[0], 100.0, position_2d[1]])
    ray_direction = -up_dir  # Downward

    closest_t = float("inf")
    closest_point = None

    # Check intersection with all triangles
    triangles = mesh_indices.reshape(-1, 3)
    for tri_indices in triangles:
        v0, v1, v2 = mesh_vertices[tri_indices]

        result = ray_triangle_intersection(ray_origin, ray_direction, v0, v1, v2)
        if result is not None:
            t, point = result
            if t < closest_t:
                closest_t = t
                closest_point = point

    return closest_point


def check_object_collision(
    obj_bbox_min: np.ndarray,
    obj_bbox_max: np.ndarray,
    other_objects: List[Tuple[np.ndarray, np.ndarray]],
) -> bool:
    """
    Check if object bounding box collides with any other objects

    Args:
        obj_bbox_min: Object bounding box minimum
        obj_bbox_max: Object bounding box maximum
        other_objects: List of (bbox_min, bbox_max) tuples for other objects

    Returns:
        True if collision detected
    """
    for other_min, other_max in other_objects:
        # AABB collision test
        if (
            obj_bbox_min[0] <= other_max[0]
            and obj_bbox_max[0] >= other_min[0]
            and obj_bbox_min[1] <= other_max[1]
            and obj_bbox_max[1] >= other_min[1]
            and obj_bbox_min[2] <= other_max[2]
            and obj_bbox_max[2] >= other_min[2]
        ):
            return True

    return False


def sample_non_colliding_position(
    background_bbox: dict,
    object_bbox_size: np.ndarray,
    existing_objects: List[dict],
    max_attempts: int = 100,
    margin: float = 0.1,
) -> Optional[np.ndarray]:
    """
    Sample a random position that doesn't collide with existing objects

    Args:
        background_bbox: Background bounding box dict with 'min' and 'max'
        object_bbox_size: Size of object to place
        existing_objects: List of existing object bounding boxes
        max_attempts: Maximum sampling attempts
        margin: Safety margin around objects

    Returns:
        Valid position or None if no valid position found
    """
    bg_min = background_bbox["min"]
    bg_max = background_bbox["max"]

    # Build list of existing bboxes
    other_bboxes = []
    for obj in existing_objects:
        if "min" in obj and "max" in obj:
            # Add margin
            expanded_min = obj["min"] - margin
            expanded_max = obj["max"] + margin
            other_bboxes.append((expanded_min, expanded_max))

    for attempt in range(max_attempts):
        # Random position
        x = np.random.uniform(bg_min[0], bg_max[0])
        z = np.random.uniform(bg_min[2], bg_max[2])
        y = bg_max[1]  # On top of background

        position = np.array([x, y, z])

        # Calculate object bbox at this position
        half_size = object_bbox_size / 2
        obj_min = position - half_size
        obj_max = position + half_size

        # Check collision
        if not check_object_collision(obj_min, obj_max, other_bboxes):
            return position

    return None  # Failed to find valid position


def compute_depth_from_camera(
    vertices: np.ndarray, view_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute depth values for vertices from camera viewpoint

    Args:
        vertices: Nx3 or Nx4 array of vertex positions
        view_matrix: 4x4 view matrix

    Returns:
        N-length array of depth values (distance from camera)
    """
    # Convert to homogeneous coordinates if needed
    if vertices.shape[1] == 3:
        vertices_h = np.hstack([vertices, np.ones((len(vertices), 1))])
    else:
        vertices_h = vertices

    # Transform to view space
    view_space = (view_matrix @ vertices_h.T).T

    # Depth is negative Z in view space (OpenGL convention)
    depths = -view_space[:, 2]

    return depths

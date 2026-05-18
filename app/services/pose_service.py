import numpy as np

from app.models.response import Point3D


def polygon_to_3d(
    polygon: list[tuple[float, float]],
    depth_map: np.ndarray,
    camera_matrix: np.ndarray,
    T_WC: np.ndarray,
) -> list[Point3D]:
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    H, W = depth_map.shape
    result: list[Point3D] = []

    for u, v in polygon:
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        depth = float(depth_map[vi, ui])
        if depth <= 0:
            continue

        x_c = (ui - cx) * depth / fx
        y_c = (vi - cy) * depth / fy
        z_c = depth

        p_c = np.array([x_c, y_c, z_c, 1.0])
        p_w = T_WC @ p_c

        result.append(Point3D(x=float(p_w[0]), y=float(p_w[1]), z=float(p_w[2])))

    return result

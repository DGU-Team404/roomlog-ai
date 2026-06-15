import gc
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

_DEPTH_W = 256
_DEPTH_H = 192
_MAX_DEPTH = 20.0

_CFG = {
    "voxel_size": 0.015,
    "sdf_trunc": 0.05,
    "confidence": 2,
}


def _get_intrinsics(K_rgb: np.ndarray) -> o3d.camera.PinholeCameraIntrinsic:
    sx, sy = _DEPTH_W / 1920.0, _DEPTH_H / 1440.0
    return o3d.camera.PinholeCameraIntrinsic(
        width=_DEPTH_W,
        height=_DEPTH_H,
        fx=K_rgb[0, 0] * sx,
        fy=K_rgb[1, 1] * sy,
        cx=K_rgb[0, 2] * sx,
        cy=K_rgb[1, 2] * sy,
    )


def _load_poses(odometry_path: Path) -> list[np.ndarray]:
    data = np.loadtxt(odometry_path, delimiter=",", skiprows=1)
    poses = []
    for row in data:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(row[5:]).as_matrix()
        T[:3, 3] = row[2:5]
        poses.append(T)
    return poses


def _load_depth(depth_path: Path, conf_path: Path) -> np.ndarray:
    with Image.open(str(depth_path)) as img:
        depth_m = np.array(img).astype(np.float32) / 1000.0
    if conf_path.exists():
        with Image.open(str(conf_path)) as img:
            conf = np.array(img)
        depth_m[conf < _CFG["confidence"]] = 0.0
    return depth_m


def run_reconstruction(
    video_path: Path,
    depth_dir: Path,
    conf_dir: Path,
    camera_matrix_path: Path,
    odometry_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    K = np.loadtxt(camera_matrix_path, delimiter=",")
    poses = _load_poses(odometry_path)
    intrinsic = _get_intrinsics(K)
    depth_files = sorted(depth_dir.glob("*.png"))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=_CFG["voxel_size"],
        sdf_trunc=_CFG["sdf_trunc"],
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    cap = cv2.VideoCapture(str(video_path))
    try:
        for i, T_WC in enumerate(poses):
            if i >= len(depth_files):
                break
            ret, bgr = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_small = np.asarray(
                Image.fromarray(rgb).resize((_DEPTH_W, _DEPTH_H), resample=Image.BILINEAR)
            )
            depth_m = _load_depth(depth_files[i], conf_dir / f"{i:06d}.png")
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_small),
                o3d.geometry.Image(depth_m),
                depth_scale=1.0,
                depth_trunc=_MAX_DEPTH,
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, intrinsic, np.linalg.inv(T_WC))
            del rgbd
    finally:
        cap.release()

    mesh = volume.extract_triangle_mesh()
    del volume
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_uniformly(number_of_points=500_000)

    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "mesh.ply"
    pcd_path = output_dir / "point_cloud.ply"

    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    del mesh, pcd
    gc.collect()

    return pcd_path, mesh_path

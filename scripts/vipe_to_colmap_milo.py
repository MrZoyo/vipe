#!/usr/bin/env python3
"""Convert ViPE reconstruction outputs into the COLMAP layout expected by MILO.

Output structure (per sequence):
<output>/<sequence>/
    images/        # extracted RGB frames
    sparse/0/
        cameras.txt / cameras.bin
        images.txt  / images.bin
        points3D.txt / points3D.bin
"""

import argparse
import logging
import struct
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import imageio
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from vipe.slam.interface import SLAMMap
from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# COLMAP camera model ids from colmap/src/base/camera_models.h
CAMERA_MODEL_NAME_TO_ID = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
}


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion ordered as (w, x, y, z)."""
    rotation = Rotation.from_matrix(matrix[:3, :3])
    quat_xyzw = rotation.as_quat()  # [x, y, z, w]
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def matrix_to_colmap_pose(c2w_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert camera-to-world matrix to COLMAP world-to-camera pose."""
    w2c = np.linalg.inv(c2w_matrix)
    quaternion = quaternion_from_matrix(w2c)
    translation = w2c[:3, 3]
    return quaternion, translation


def ensure_dirs(output_root: Path) -> Tuple[Path, Path]:
    """Create MILO COLMAP folder layout and return (images_dir, sparse0_dir)."""
    images_dir = output_root / "images"
    sparse_dir = output_root / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, sparse_dir


def extract_frames(artifact: ArtifactPath, images_dir: Path) -> Tuple[int, int]:
    """Extract RGB video frames into the images/ folder."""
    logger.info(f"Extracting frames from {artifact.rgb_path}")
    last_height, last_width = 0, 0
    for frame_idx, rgb in read_rgb_artifacts(artifact.rgb_path):
        frame_height, frame_width = rgb.shape[:2]
        last_height, last_width = frame_height, frame_width
        img_path = images_dir / f"{frame_idx:06d}.jpg"
        imageio.imwrite(str(img_path), (rgb.cpu().numpy() * 255).astype(np.uint8))
        if frame_idx % 50 == 0:
            logger.info(f"Extracted frame {frame_idx}")
    if last_width == 0 or last_height == 0:
        raise ValueError(f"No frames extracted from {artifact.rgb_path}")

    logger.info(f"Finished extracting {artifact.rgb_path}")
    return last_width, last_height


def write_cameras(sparse_dir: Path, frame_width: int, frame_height: int, artifact: ArtifactPath):
    """
    Write COLMAP cameras.txt and cameras.bin using PINHOLE model (fx, fy, cx, cy).
    MILO loader only accepts undistorted PINHOLE/SIMPLE_PINHOLE.
    """
    _, intrinsics, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)
    camera_type = camera_types[0]
    assert camera_type == CameraType.PINHOLE, "MILO exporter currently supports PINHOLE cameras only."
    fx, fy, cx, cy = intrinsics[0].cpu().numpy()
    camera_id = 1

    cameras_txt = sparse_dir / "cameras.txt"
    with cameras_txt.open("w") as f_txt:
        f_txt.write(f"{camera_id} PINHOLE {frame_width} {frame_height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    params = [float(fx), float(fy), float(cx), float(cy)]
    model_id = CAMERA_MODEL_NAME_TO_ID["PINHOLE"]
    cameras_bin = sparse_dir / "cameras.bin"
    with cameras_bin.open("wb") as f_bin:
        f_bin.write(struct.pack("<Q", 1))  # number of cameras
        f_bin.write(struct.pack("<I", camera_id))
        f_bin.write(struct.pack("<i", model_id))
        f_bin.write(struct.pack("<QQ", frame_width, frame_height))
        for p in params:
            f_bin.write(struct.pack("<d", float(p)))

    logger.info(
        f"Written cameras for frame size {frame_width}x{frame_height}, fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}"
    )


def write_images(sparse_dir: Path, artifact: ArtifactPath) -> Tuple[int, List[str]]:
    """Write COLMAP images.txt and images.bin without 2D-3D tracks."""
    pose_npz = np.load(artifact.pose_path)
    poses = pose_npz["data"]
    indices = pose_npz["inds"]
    images_txt = sparse_dir / "images.txt"
    image_names: List[str] = []

    with images_txt.open("w") as f_txt, (sparse_dir / "images.bin").open("wb") as f_bin:
        f_bin.write(struct.pack("<Q", len(poses)))  # number of images
        for image_id, (pose_matrix, frame_idx) in enumerate(zip(poses, indices), start=1):
            quaternion, translation = matrix_to_colmap_pose(pose_matrix)
            qw, qx, qy, qz = quaternion
            tx, ty, tz = translation
            image_name = f"{frame_idx:06d}.jpg"
            image_names.append(image_name)

            # Text file
            f_txt.write(
                f"{image_id} {qw:.15f} {qx:.15f} {qy:.15f} {qz:.15f} {tx:.15f} {ty:.15f} {tz:.15f} 1 {image_name}\n\n"
            )

            # Binary file
            f_bin.write(struct.pack("<I", image_id))
            f_bin.write(struct.pack("<ddddddd", qw, qx, qy, qz, tx, ty, tz))
            f_bin.write(struct.pack("<I", 1))  # camera_id
            name_bytes = image_name.encode("utf-8") + b"\x00"
            f_bin.write(name_bytes)
            f_bin.write(struct.pack("<Q", 0))  # num_points2D (no correspondences)

    logger.info(f"Written {len(poses)} image poses")
    return len(poses), image_names


def collect_points_from_slam_map(artifact: ArtifactPath) -> Iterable[Tuple[int, float, float, float, int, int, int]]:
    """Yield point tuples from SLAM map: (id, x, y, z, r, g, b)."""
    slam_map = SLAMMap.load(artifact.slam_map_path, device=torch.device("cpu"))
    point_id = 1
    for keyframe_idx, _ in enumerate(slam_map.dense_disp_frame_inds):
        xyz, rgb = slam_map.get_dense_disp_pcd(keyframe_idx)
        xyz = xyz.cpu().numpy()
        rgb = rgb.cpu().numpy()
        for xyz_pt, rgb_pt in zip(xyz, rgb):
            r, g, b = (rgb_pt * 255).astype(np.uint8)
            yield (point_id, float(xyz_pt[0]), float(xyz_pt[1]), float(xyz_pt[2]), int(r), int(g), int(b))
            point_id += 1


def collect_points_from_depth(
    artifact: ArtifactPath, images_dir: Path, depth_step: int, spatial_subsample: int = 4
) -> Iterable[Tuple[int, float, float, float, int, int, int]]:
    """
    Produce 3D points by unprojecting depth maps.
    Returns an iterator of tuples: (id, x, y, z, r, g, b).
    """
    pose_inds, pose_data = read_pose_artifacts(artifact.pose_path)
    pose_map = {int(idx): pose for idx, pose in zip(pose_inds, pose_data)}

    intr_inds, intrinsics, camera_types = read_intrinsics_artifacts(artifact.intrinsics_path)
    intr_map = {int(idx): intr for idx, intr in zip(intr_inds, intrinsics)}
    camera_type = camera_types[0]
    rays: np.ndarray | None = None
    point_id = 1

    for depth_iter_idx, (frame_idx, depth) in enumerate(read_depth_artifacts(artifact.depth_path)):
        if depth_iter_idx % depth_step != 0:
            continue

        pose = pose_map.get(frame_idx, None)
        intr = intr_map.get(frame_idx, None)
        if pose is None or intr is None:
            logger.warning("Skipping frame %s due to missing pose or intrinsics", frame_idx)
            continue

        rgb_path = images_dir / f"{frame_idx:06d}.jpg"
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            logger.warning("RGB frame missing for depth frame %s at %s", frame_idx, rgb_path)
            continue
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = rgb.shape[:2]
        rgb = rgb[::spatial_subsample, ::spatial_subsample]

        if rays is None:
            camera_model = camera_type.build_camera_model(intr)
            disp_v, disp_u = torch.meshgrid(
                torch.arange(frame_height).float()[::spatial_subsample],
                torch.arange(frame_width).float()[::spatial_subsample],
                indexing="ij",
            )
            if camera_type == CameraType.PANORAMA:
                disp_v = disp_v / (frame_height - 1)
                disp_u = disp_u / (frame_width - 1)
            disp = torch.ones_like(disp_v)
            pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
            rays = pts[..., :3].numpy()
            if camera_type != CameraType.PANORAMA:
                rays /= rays[..., 2:3]

        if depth is None:
            continue

        pcd = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
        depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
        rgb_masked, pcd_masked = rgb[depth_mask], pcd[depth_mask]
        c2w_matrix = pose.matrix().numpy()
        pcd_world = pcd_masked @ c2w_matrix[:3, :3].T + c2w_matrix[:3, 3][None]

        for rgb_pt, xyz_pt in zip(rgb_masked, pcd_world):
            yield (
                point_id,
                float(xyz_pt[0]),
                float(xyz_pt[1]),
                float(xyz_pt[2]),
                int(rgb_pt[0]),
                int(rgb_pt[1]),
                int(rgb_pt[2]),
            )
            point_id += 1


def write_points3d(sparse_dir: Path, points: Iterable[Tuple[int, float, float, float, int, int, int]]):
    """Write points3D in both txt and bin formats."""
    txt_path = sparse_dir / "points3D.txt"
    bin_path = sparse_dir / "points3D.bin"

    # Materialize because we need the count twice.
    points_list = list(points)
    with txt_path.open("w") as f_txt:
        f_txt.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        for pid, x, y, z, r, g, b in points_list:
            f_txt.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")

    with bin_path.open("wb") as f_bin:
        f_bin.write(struct.pack("<Q", len(points_list)))
        for pid, x, y, z, r, g, b in points_list:
            f_bin.write(struct.pack("<Q", pid))
            f_bin.write(struct.pack("<ddd", x, y, z))
            f_bin.write(struct.pack("<BBB", r, g, b))
            f_bin.write(struct.pack("<d", 0.0))  # reprojection error placeholder
            f_bin.write(struct.pack("<Q", 0))  # empty track length

    logger.info(f"Written {len(points_list)} 3D points")


def convert_for_milo(artifact: ArtifactPath, output_root: Path, depth_step: int, use_slam_map: bool):
    logger.info(f"Converting {artifact.artifact_name} to MILO COLMAP at {output_root}")
    images_dir, sparse_dir = ensure_dirs(output_root)

    required_files = [artifact.rgb_path, artifact.pose_path, artifact.intrinsics_path]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    # Extract frames before writing pose/image metadata.
    frame_width, frame_height = extract_frames(artifact, images_dir)

    write_cameras(sparse_dir, frame_width, frame_height, artifact)
    write_images(sparse_dir, artifact)

    if use_slam_map:
        if not artifact.slam_map_path.exists():
            raise FileNotFoundError(f"SLAM map not found: {artifact.slam_map_path}")
        points_iter = collect_points_from_slam_map(artifact)
    else:
        if not Path(artifact.depth_path).exists():
            raise FileNotFoundError(f"Depth artifact not found: {artifact.depth_path}")
        points_iter = collect_points_from_depth(artifact, images_dir, depth_step)

    write_points3d(sparse_dir, points_iter)
    logger.info(f"Finished MILO-format export: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ViPE results to MILO-compatible COLMAP format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("vipe_path", type=Path, help="Path to ViPE results directory")
    parser.add_argument("--sequence", "-s", type=str, default=None, help="Sequence name (mp4 stem) to convert only")
    parser.add_argument("--depth_step", type=int, default=16, help="Depth sampling stride for point cloud")
    parser.add_argument("--use_slam_map", action="store_true", help="Use SLAM map instead of unprojecting depth")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output root (default: <vipe_path>_milo_colmap)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.vipe_path.exists():
        logger.error("ViPE path does not exist: %s", args.vipe_path)
        return 1

    artifacts = list(ArtifactPath.glob_artifacts(args.vipe_path, use_video=True))
    if args.sequence is not None:
        artifacts = [a for a in artifacts if a.artifact_name == args.sequence]
    if not artifacts:
        logger.error("No matching artifacts found under %s", args.vipe_path)
        return 1

    if args.output is None:
        args.output = args.vipe_path.parent / f"{args.vipe_path.name}_milo"

    for artifact in artifacts:
        convert_for_milo(artifact, args.output / artifact.artifact_name, args.depth_step, args.use_slam_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

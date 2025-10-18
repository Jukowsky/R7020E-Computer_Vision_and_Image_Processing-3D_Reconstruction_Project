import re
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:
    raise SystemExit("OpenCV (cv2) is required. Install with: pip install opencv-python") from e

try:
    import open3d as o3d
except Exception as e:
    raise SystemExit("open3d is required. Install with: pip install open3d") from e

try:
    from scipy.spatial import ConvexHull
except Exception as e:
    raise SystemExit("scipy is required for convex hull volume. Install with: pip install scipy") from e


# ---- USER PATHS ----
DIR_COLOR_INFO = r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_color_camera_info"
DIR_DEPTH_IMG  = r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_depth_image_raw"
DIR_COLORED_PCD = r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\depth_registered_colored_pointclouds"

# Write results to a folder you fully own (avoid Downloads lock):
OUT_ROOT = Path.home() / "Documents" / "Project2" / "outputs_step4_batch"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Unique CSV name per run (prevents “file in use” errors)
CSV_NAME = f"batch_volumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
DEFAULT_DEPTH_SCALE = 0.001  # mm -> m fallback


# ---- HELPERS ----
def extract_timestamp(s: str) -> Optional[int]:
    nums = re.findall(r'\d+', s)
    if not nums:
        return None
    return int(nums[-1])

def index_dir_by_ts(dir_path: str, exts: Tuple[str, ...]) -> Dict[int, Path]:
    d = Path(dir_path)
    idx: Dict[int, Path] = {}
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            ts = extract_timestamp(p.name)
            if ts is not None:
                idx[ts] = p
    return idx

def parse_camera_info_txt(p: Path) -> Dict:
    import re as _re
    out = {"K": None, "height": None, "width": None}
    text = p.read_text(encoding='utf-8', errors='ignore')
    mK = _re.search(r'K:\s*\[([^\]]+)\]', text)
    if mK:
        vals = [float(x.strip()) for x in mK.group(1).split(',')]
        if len(vals) == 9:
            out['K'] = vals
    mH = _re.search(r'height:\s*([0-9]+)', text)
    mW = _re.search(r'width:\s*([0-9]+)', text)
    if mH: out['height'] = int(mH.group(1))
    if mW: out['width'] = int(mW.group(1))
    return out

def nearest_ts(target: int, candidates: List[int]) -> Optional[int]:
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t - target))

def estimate_depth_scale_from_pcd(depth_img: np.ndarray, colored_pcd_path: Path) -> Optional[float]:
    try:
        pcd = o3d.io.read_point_cloud(str(colored_pcd_path))
        if pcd.is_empty():
            return None
        pts = np.asarray(pcd.points)
        z_med = float(np.median(pts[:,2]))
        nonzero = np.argwhere(depth_img > 0)
        if nonzero.size == 0:
            return None
        sample_idxs = nonzero[::max(1, len(nonzero)//200)]
        raw_vals = depth_img[sample_idxs[:,0], sample_idxs[:,1]].astype(np.float64)
        raw_med = float(np.median(raw_vals))
        if raw_med <= 0:
            return None
        return z_med / raw_med
    except Exception:
        return None

def depth_to_pointcloud(depth_img: np.ndarray, K: List[float], depth_scale: float) -> o3d.geometry.PointCloud:
    fx, _, cx, _, fy, cy, _, _, _ = K
    H, W = depth_img.shape[:2]
    us = np.arange(W, dtype=np.float32)
    vs = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    Z = depth_img.astype(np.float32) * depth_scale
    valid = Z > 0
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd

def segment_and_volume(pcd: o3d.geometry.PointCloud) -> Dict:
    VOXEL_SIZE = 0.005
    PLANE_DISTANCE_THRESH = 0.01
    PLANE_RANSAC_N = 3
    PLANE_NUM_ITERS = 1000
    DBSCAN_EPS = 0.02
    DBSCAN_MIN_POINTS = 50

    pcd_ds = pcd.voxel_down_sample(VOXEL_SIZE)
    plane_model, inliers = pcd_ds.segment_plane(distance_threshold=PLANE_DISTANCE_THRESH,
                                                ransac_n=PLANE_RANSAC_N,
                                                num_iterations=PLANE_NUM_ITERS)
    plane_cloud = pcd_ds.select_by_index(inliers)
    obj_cloud = pcd_ds.select_by_index(inliers, invert=True)

    labels = np.array(obj_cloud.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        obj_cluster = obj_cloud
    else:
        largest_label = int(np.argmax([np.sum(labels == i) for i in range(labels.max() + 1)]))
        idx = np.where(labels == largest_label)[0].tolist()
        obj_cluster = obj_cloud.select_by_index(idx)

    pts = np.asarray(obj_cluster.points)
    aabb = obj_cluster.get_axis_aligned_bounding_box()
    obb = obj_cluster.get_oriented_bounding_box()
    aabb_ext = aabb.get_extent()
    obb_ext = obb.extent
    vol_aabb = float(aabb_ext[0] * aabb_ext[1] * aabb_ext[2])
    vol_obb  = float(obb_ext[0] * obb_ext[1] * obb_ext[2])

    if pts.shape[0] >= 4:
        vol_ch = float(ConvexHull(pts).volume)
    else:
        vol_ch = 0.0

    return {
        "counts": {
            "total": len(pcd.points),
            "downsampled": len(pcd_ds.points),
            "after_plane": len(obj_cloud.points),
            "object_cluster": len(obj_cluster.points)
        },
        "volumes_m3": {
            "aabb": vol_aabb,
            "obb": vol_obb,
            "convex_hull": vol_ch
        },
        "clouds": {
            "plane": plane_cloud,
            "without_plane": obj_cloud,
            "object": obj_cluster
        }
    }


def safe_open_csv(path: Path):
    """
    Try to open CSV for writing; if locked, create a new name with timestamp.
    """
    try:
        f = path.open('w', newline='', encoding='utf-8')
        return f, path
    except PermissionError:
        alt = path.with_name(path.stem + f"_{int(time.time())}" + path.suffix)
        f = alt.open('w', newline='', encoding='utf-8')
        return f, alt


def main():
    # Index
    idx_color_info = index_dir_by_ts(DIR_COLOR_INFO, ('.txt', '.yaml', '.yml'))
    idx_depth_img  = index_dir_by_ts(DIR_DEPTH_IMG,  ('.png',))
    idx_col_pcd    = index_dir_by_ts(DIR_COLORED_PCD, ('.pcd',))

    if not idx_color_info:
        raise SystemExit("No camera info files found.")
    any_cinfo = next(iter(sorted(idx_color_info.items())))[1]
    K = parse_camera_info_txt(any_cinfo).get("K")
    if not K:
        raise SystemExit("Failed to parse K from any camera info file.")

    out_csv = OUT_ROOT / CSV_NAME
    f, actual_csv = safe_open_csv(out_csv)
    with f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "depth_path", "colored_pcd_used", "depth_scale_m_per_unit",
                         "points_total", "points_downsampled", "points_after_plane",
                         "points_object_cluster", "vol_aabb_m3", "vol_obb_m3", "vol_convex_hull_m3"])

        for ts, depth_path in sorted(idx_depth_img.items()):
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                print(f"[Skip] Could not read depth: {depth_path}")
                continue
            # Nearest colored pcd to estimate scale (optional)
            ts_colpcd = nearest_ts(ts, list(idx_col_pcd.keys()))
            colpcd_path = idx_col_pcd.get(ts_colpcd) if ts_colpcd is not None else None
            if colpcd_path:
                scale = estimate_depth_scale_from_pcd(depth, colpcd_path) or DEFAULT_DEPTH_SCALE
            else:
                scale = DEFAULT_DEPTH_SCALE

            # Reconstruct -> segment -> volume
            pcd = depth_to_pointcloud(depth, K, scale)
            res = segment_and_volume(pcd)

            # Save clouds
            out_dir = OUT_ROOT / f"{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(out_dir / "plane.pcd"), res["clouds"]["plane"], write_ascii=True)
            o3d.io.write_point_cloud(str(out_dir / "without_plane.pcd"), res["clouds"]["without_plane"], write_ascii=True)
            o3d.io.write_point_cloud(str(out_dir / "object.pcd"), res["clouds"]["object"], write_ascii=True)

            writer.writerow([
                ts,
                str(depth_path),
                str(colpcd_path) if colpcd_path else "",
                scale,
                res["counts"]["total"],
                res["counts"]["downsampled"],
                res["counts"]["after_plane"],
                res["counts"]["object_cluster"],
                res["volumes_m3"]["aabb"],
                res["volumes_m3"]["obb"],
                res["volumes_m3"]["convex_hull"],
            ])

    print(f"Wrote CSV: {actual_csv}")
    print(f"Per-frame outputs under: {OUT_ROOT}\\<timestamp>\\")


if __name__ == "__main__":
    main()

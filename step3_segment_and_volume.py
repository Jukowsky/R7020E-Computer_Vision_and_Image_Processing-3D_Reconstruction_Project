
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise SystemExit("open3d is required. Install with: pip install open3d") from e

try:
    from scipy.spatial import ConvexHull
except Exception as e:
    raise SystemExit("scipy is required for convex hull volume. Install with: pip install scipy") from e

# ---- USER CONFIG ----
# If left None, we try to auto-pick the most recent reconstructed PCD from outputs_step2
INPUT_PCD: Optional[str] = None  # e.g., r"C:\path\to\outputs_step2\reconstructed_from_depth_....pcd"

# Preprocessing

# Preprocessing
VOXEL_SIZE = 0.004    # was 0.005

# Plane removal (RANSAC)
PLANE_DISTANCE_THRESH = 0.003  # was 0.01 (keep more near the table)

# Clustering
DBSCAN_EPS = 0.03     # was 0.02 (fuse nearby chunks)
DBSCAN_MIN_POINTS = 30  # was 50 (keep medium clusters)




# Plane removal (RANSAC)
PLANE_RANSAC_N = 3
PLANE_NUM_ITERS = 1000
# Clustering

OUT_DIR = Path.cwd() / "outputs_step3"
OUT_DIR.mkdir(exist_ok=True, parents=True)

def find_latest_reconstructed_pcd(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    cands = sorted([p for p in dir_path.glob("reconstructed_from_depth_*.pcd")], key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def voxel_downsample(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    if voxel and voxel > 0:
        return pcd.voxel_down_sample(voxel_size=voxel)
    return pcd

def remove_plane(pcd: o3d.geometry.PointCloud, dist_thresh: float, ransac_n: int, num_iters: int):
    if len(pcd.points) == 0:
        return pcd, o3d.geometry.PointCloud()
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=ransac_n, num_iterations=num_iters)
    plane_cloud = pcd.select_by_index(inliers)
    obj_cloud = pcd.select_by_index(inliers, invert=True)
    print(f"Plane model: {plane_model}, inliers: {len(inliers)}, remaining: {len(obj_cloud.points)}")
    return obj_cloud, plane_cloud

def cluster_largest(pcd: o3d.geometry.PointCloud, eps: float, min_points: int) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return pcd
    largest_label = int(np.argmax([np.sum(labels == i) for i in range(labels.max() + 1)]))
    idx = np.where(labels == largest_label)[0].tolist()
    return pcd.select_by_index(idx)

def convex_hull_volume(points: np.ndarray) -> float:
    if points.shape[0] < 4:
        return 0.0
    hull = ConvexHull(points)
    return float(hull.volume)

def main():
    # Resolve input PCD
    p_in = Path(INPUT_PCD) if INPUT_PCD else find_latest_reconstructed_pcd(Path.cwd() / "outputs_step2")
    if p_in is None or not p_in.exists():
        raise SystemExit("Could not find input PCD. Set INPUT_PCD or run step 2 first.")
    print(f"Input PCD: {p_in}")

    pcd = o3d.io.read_point_cloud(str(p_in))
    print(f"Loaded {len(pcd.points)} points.")

    # Downsample
    pcd_ds = voxel_downsample(pcd, VOXEL_SIZE)
    print(f"Downsampled to {len(pcd_ds.points)} points (voxel={VOXEL_SIZE} m).")

    # Remove dominant plane (table/floor)
    obj_cloud, plane_cloud = remove_plane(pcd_ds, PLANE_DISTANCE_THRESH, PLANE_RANSAC_N, PLANE_NUM_ITERS)

    # Cluster and take largest cluster as object
    obj_cluster = cluster_largest(obj_cloud, DBSCAN_EPS, DBSCAN_MIN_POINTS)
    print(f"Object cluster points: {len(obj_cluster.points)}")

    # Save intermediates
    out_pcd_plane = OUT_DIR / f"step3_plane_inliers.pcd"
    out_pcd_noplane = OUT_DIR / f"step3_without_plane.pcd"
    out_pcd_object = OUT_DIR / f"step3_object_cluster.pcd"
    if len(plane_cloud.points) > 0:
        o3d.io.write_point_cloud(str(out_pcd_plane), plane_cloud, write_ascii=True)
    o3d.io.write_point_cloud(str(out_pcd_noplane), obj_cloud, write_ascii=True)
    o3d.io.write_point_cloud(str(out_pcd_object), obj_cluster, write_ascii=True)
    print(f"Saved: {out_pcd_object}")

    # Volumes
    pts = np.asarray(obj_cluster.points)
    aabb = obj_cluster.get_axis_aligned_bounding_box()
    obb = obj_cluster.get_oriented_bounding_box()
    aabb_extent = aabb.get_extent()
    obb_extent = obb.extent
    vol_aabb = float(aabb_extent[0] * aabb_extent[1] * aabb_extent[2])
    vol_obb  = float(obb_extent[0] * obb_extent[1] * obb_extent[2])
    vol_ch   = convex_hull_volume(pts)  # convex hull volume in m^3

    report = {
        "input_pcd": str(p_in),
        "points_total": len(pcd.points),
        "points_downsampled": len(pcd_ds.points),
        "points_after_plane_removal": len(obj_cloud.points),
        "points_object_cluster": len(obj_cluster.points),
        "voxel_size_m": VOXEL_SIZE,
        "plane_removal": {
            "distance_threshold_m": PLANE_DISTANCE_THRESH,
            "ransac_n": PLANE_RANSAC_N,
            "num_iterations": PLANE_NUM_ITERS
        },
        "clustering": {
            "dbscan_eps_m": DBSCAN_EPS,
            "dbscan_min_points": DBSCAN_MIN_POINTS
        },
        "volumes_m3": {
            "axis_aligned_bounding_box": vol_aabb,
            "oriented_bounding_box": vol_obb,
            "convex_hull": vol_ch
        },
        "aabb_extent_m": aabb_extent.tolist(),
        "obb_extent_m": obb_extent.tolist()
    }
    out_json = OUT_DIR / "step3_volume_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote report: {out_json}")

if __name__ == "__main__":
    main()

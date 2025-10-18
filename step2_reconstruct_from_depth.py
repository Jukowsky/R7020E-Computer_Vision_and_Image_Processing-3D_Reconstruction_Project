
import json
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except Exception as e:
    raise SystemExit("OpenCV (cv2) is required. Install with: pip install opencv-python") from e

try:
    import open3d as o3d
except Exception as e:
    raise SystemExit("open3d is required. Install with: pip install open3d") from e

# ---- USER PATHS (edit if needed) ----
DIR_COLOR_INFO = r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_color_camera_info"
DIR_DEPTH_INFO = r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_depth_camera_info"
DIR_COLOR_IMG  = r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_color_image_raw"
DIR_DEPTH_IMG  = r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_depth_image_raw"
DIR_COLORED_PCD = r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\depth_registered_colored_pointclouds"

OUT_DIR = Path.cwd() / "outputs_step2"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---- HELPERS ----

def extract_timestamp(s: str) -> Optional[int]:
    # Grab the last long integer in the filename (nanoseconds-like)
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
    # Very light YAML-like parser for K (intrinsics) and image size
    out = {"K": None, "height": None, "width": None, "frame_id": None, "distortion_model": None, "D": None}
    text = p.read_text(encoding='utf-8', errors='ignore')
    # K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    mK = re.search(r'K:\s*\[([^\]]+)\]', text)
    if mK:
        vals = [float(x.strip()) for x in mK.group(1).split(',')]
        if len(vals) == 9:
            out['K'] = vals
    mH = re.search(r'height:\s*([0-9]+)', text)
    mW = re.search(r'width:\s*([0-9]+)', text)
    if mH: out['height'] = int(mH.group(1))
    if mW: out['width'] = int(mW.group(1))
    mF = re.search(r'frame_id:\s*"([^"]+)"', text)
    if mF: out['frame_id'] = mF.group(1)
    mDM = re.search(r'distortion_model:\s*"([^"]+)"', text)
    if mDM: out['distortion_model'] = mDM.group(1)
    mD = re.search(r'D:\s*\[([^\]]+)\]', text)
    if mD:
        out['D'] = [float(x.strip()) for x in mD.group(1).split(',')]
    return out

def nearest_ts(target: int, candidates: List[int]) -> Optional[int]:
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t - target))

def estimate_depth_scale_from_pcd(depth_img: np.ndarray, K: List[float], colored_pcd_path: Path) -> Optional[float]:
    """Try a crude estimate of depth scale (units of the PNG) by comparing Z in PCD vs raw depth at random valid pixels.
    Returns scale to convert raw depth to meters (e.g., 0.001 if units are mm)."""
    try:
        pcd = o3d.io.read_point_cloud(str(colored_pcd_path))
        if pcd.is_empty():
            return None
        pts = np.asarray(pcd.points)
        # Heuristic: compute median Z of PCD (meters)
        z_med = float(np.median(pts[:,2]))
        # Sample raw depth where nonzero
        nonzero = np.argwhere(depth_img > 0)
        if nonzero.size == 0:
            return None
        sample_idxs = nonzero[::max(1, len(nonzero)//200)]  # up to ~200 samples
        raw_vals = depth_img[sample_idxs[:,0], sample_idxs[:,1]].astype(np.float64)
        raw_med = float(np.median(raw_vals))
        if raw_med <= 0:
            return None
        scale = z_med / raw_med
        return scale
    except Exception:
        return None

def depth_to_pointcloud(depth_img: np.ndarray, K: List[float], depth_scale: float) -> np.ndarray:
    """Back-project a depth image (H,W) to Nx3 point array given intrinsics and scale (meters per depth_unit)."""
    fx, _, cx, _, fy, cy, _, _, _ = K
    H, W = depth_img.shape[:2]
    us = np.arange(W, dtype=np.float32)
    vs = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    Z = depth_img.astype(np.float32) * depth_scale  # meters
    valid = Z > 0
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1)
    return pts

def main():
    # Index files
    idx_color_info = index_dir_by_ts(DIR_COLOR_INFO, ('.txt', '.yaml', '.yml'))
    idx_depth_info = index_dir_by_ts(DIR_DEPTH_INFO, ('.txt', '.yaml', '.yml'))
    idx_color_img  = index_dir_by_ts(DIR_COLOR_IMG,  ('.png',))
    idx_depth_img  = index_dir_by_ts(DIR_DEPTH_IMG,  ('.png',))
    idx_col_pcd    = index_dir_by_ts(DIR_COLORED_PCD, ('.pcd',))

    # Pick a target timestamp present in depth image (anchor)
    if not idx_depth_img:
        raise SystemExit("No depth images found.")
    target_ts = sorted(idx_depth_img.keys())[0]

    # Find nearest matches
    ts_depth = target_ts
    ts_color = nearest_ts(ts_depth, list(idx_color_img.keys()))
    ts_cinfo = nearest_ts(ts_depth, list(idx_color_info.keys()))
    ts_dinfo = nearest_ts(ts_depth, list(idx_depth_info.keys()))
    ts_colpcd = nearest_ts(ts_depth, list(idx_col_pcd.keys()))

    depth_path = idx_depth_img[ts_depth]
    color_path = idx_color_img.get(ts_color)
    cinfo_path = idx_color_info.get(ts_cinfo)
    dinfo_path = idx_depth_info.get(ts_dinfo)
    colpcd_path = idx_col_pcd.get(ts_colpcd)

    # Load intrinsics from color camera info (depth seems registered to color frame per your sample)
    if cinfo_path is None:
        raise SystemExit("No camera_color_camera_info found (timestamp-nearest).")
    ci = parse_camera_info_txt(cinfo_path)
    if not ci.get("K"):
        raise SystemExit(f"Failed to parse K from {cinfo_path}")
    K = ci["K"]
    print(f"Using intrinsics from: {cinfo_path.name}")
    print(f"K = {K} (fx,0,cx, 0,fy,cy, 0,0,1)")
    print(f"Frame ID: {ci.get('frame_id')}  Distortion model: {ci.get('distortion_model')}")

    # Load depth image (uint16 expected)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit(f"Failed to read depth image: {depth_path}")
    print(f"Depth image: {depth_path.name}  shape={depth.shape} dtype={depth.dtype}")

    # Estimate depth scale (meters per unit). Try via colored PCD median as heuristic, else assume millimeters.
    depth_scale = None
    if colpcd_path is not None:
        print(f"Trying to estimate depth scale using colored PCD: {colpcd_path.name}")
        depth_scale = estimate_depth_scale_from_pcd(depth, K, colpcd_path)

    if depth_scale is None:
        depth_scale = 0.001  # assume millimeters -> meters
        print("Depth scale not estimated from PCD. Assuming 0.001 m/unit (mm).")

    print(f"Depth scale: {depth_scale:.6f} m per unit")

    # Back-project to point cloud
    pts = depth_to_pointcloud(depth, K, depth_scale)
    print(f"Generated {len(pts)} 3D points from depth.")

    # Save to PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    out_pcd = OUT_DIR / f"reconstructed_from_depth_{ts_depth}.pcd"
    o3d.io.write_point_cloud(str(out_pcd), pcd, write_ascii=True)
    print(f"Saved PCD: {out_pcd}")

    # Log chosen files
    log = {
        "chosen_timestamps": {
            "depth_img": ts_depth,
            "color_img": ts_color,
            "color_info": ts_cinfo,
            "depth_info": ts_dinfo,
            "colored_pcd": ts_colpcd
        },
        "paths": {
            "depth_img": str(depth_path),
            "color_img": str(color_path) if color_path else None,
            "color_info": str(cinfo_path) if cinfo_path else None,
            "depth_info": str(dinfo_path) if dinfo_path else None,
            "colored_pcd": str(colpcd_path) if colpcd_path else None,
            "output_pcd": str(out_pcd)
        },
        "K": K,
        "depth_scale_m_per_unit": depth_scale
    }
    out_log = OUT_DIR / f"step2_selection_{ts_depth}.json"
    out_log.write_text(json.dumps(log, indent=2))
    print(f"Wrote log: {out_log}")

if __name__ == "__main__":
    main()

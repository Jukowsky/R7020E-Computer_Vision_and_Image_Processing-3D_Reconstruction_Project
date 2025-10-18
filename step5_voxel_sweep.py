import json
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise SystemExit("open3d is required. Install with: pip install open3d") from e

try:
    from scipy.ndimage import binary_dilation, binary_erosion, label
except Exception as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e


# =========================
# USER CONFIG
# =========================
INPUT_OBJECT_PCD: Optional[str] = None   # e.g., r"C:\\...\\outputs_step3\\step3_object_cluster.pcd"

VOXEL_SIZE = 0.006       # was 0.008 (6 mm)
PADDING_VOXELS = 5       # was 3
DILATE_RINGS = 2         # was 1
CLOSING_ITERS = 2        # was 1
REMOVE_SMALL_COMPONENTS = True
MIN_COMPONENT_VOXELS = 300  # was 500 (avoid deleting thin connections)

OUT_DIR = Path.cwd() / "outputs_voxel_solid"
# =========================


def find_latest_object_pcd() -> Optional[Path]:
    base = Path.cwd() / "outputs_step3"
    cands = sorted(base.glob("step3_object_cluster.pcd"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def quantize_to_grid(points: np.ndarray, voxel: float, pad: int):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dims = np.ceil((maxs - mins) / voxel).astype(int) + 1 + 2 * pad
    # map to indices with padding offset
    idx = np.floor((points - mins) / voxel).astype(int) + pad
    return mins, dims.astype(int), idx


def largest_component(mask: np.ndarray) -> np.ndarray:
    # label connected components in 3D (6-connectivity)
    structure = np.zeros((3,3,3), dtype=bool)
    structure[1,1,0] = structure[1,1,2] = True
    structure[1,0,1] = structure[1,2,1] = True
    structure[0,1,1] = structure[2,1,1] = True
    labeled, num = label(mask, structure=structure)
    if num <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background
    keep_label = sizes.argmax()
    return (labeled == keep_label)


def flood_fill_exterior(solid_shell: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of exterior air voxels reachable from the boundary.
    solid_shell=True means blocked (object shell). False means potentially air.
    """
    Z, Y, X = solid_shell.shape
    exterior = np.zeros_like(solid_shell, dtype=bool)
    q = deque()

    # seed queue with all boundary voxels that are not solid
    def try_push(z,y,x):
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            if not solid_shell[z,y,x] and not exterior[z,y,x]:
                exterior[z,y,x] = True
                q.append((z,y,x))

    for z in [0, Z-1]:
        for y in range(Y):
            for x in range(X):
                try_push(z,y,x)
    for z in range(Z):
        for y in [0, Y-1]:
            for x in range(X):
                try_push(z,y,x)
    for z in range(Z):
        for y in range(Y):
            for x in [0, X-1]:
                try_push(z,y,x)

    # 6-neighborhood BFS
    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    while q:
        z,y,x = q.popleft()
        for dz,dy,dx in nbrs:
            nz, ny, nx = z+dz, y+dy, x+dx
            if 0 <= nz < Z and 0 <= ny < Y and 0 <= nx < X:
                if not solid_shell[nz,ny,nx] and not exterior[nz,ny,nx]:
                    exterior[nz,ny,nx] = True
                    q.append((nz,ny,nx))
    return exterior


def main():
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    p_in = Path(INPUT_OBJECT_PCD) if INPUT_OBJECT_PCD else find_latest_object_pcd()
    if p_in is None or not p_in.exists():
        raise SystemExit("Could not find object PCD. Provide INPUT_OBJECT_PCD or run Step 3 first.")
    print(f"Object PCD: {p_in}")

    # Load points
    pcd = o3d.io.read_point_cloud(str(p_in))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.shape[0] < 50:
        raise SystemExit("Too few points for voxelization.")

    # Build grid
    mins, dims, idx = quantize_to_grid(pts, VOXEL_SIZE, PADDING_VOXELS)
    Z, Y, X = int(dims[2]), int(dims[1]), int(dims[0])  # careful with axis order; we’ll use (z,y,x)
    shell = np.zeros((Z, Y, X), dtype=bool)  # z,y,x
    # note: idx currently is (N,3) as (x,y,z) indices; map to (z,y,x)
    ix, iy, iz = idx[:,0], idx[:,1], idx[:,2]
    # clamp just in case
    ix = np.clip(ix, 0, X-1); iy = np.clip(iy, 0, Y-1); iz = np.clip(iz, 0, Z-1)
    shell[iz, iy, ix] = True

    # (Optional) keep largest component to drop speckles
    if REMOVE_SMALL_COMPONENTS:
        largest = largest_component(shell)
        if MIN_COMPONENT_VOXELS > 0:
            # additionally zero-out tiny components by thresholding component sizes
            structure = np.zeros((3,3,3), dtype=bool)
            structure[1,1,0] = structure[1,1,2] = True
            structure[1,0,1] = structure[1,2,1] = True
            structure[0,1,1] = structure[2,1,1] = True
            labeled, num = label(shell, structure=structure)
            counts = np.bincount(labeled.ravel())
            keep = np.zeros_like(shell, dtype=bool)
            for lbl, cnt in enumerate(counts):
                if lbl == 0:
                    continue
                if cnt >= MIN_COMPONENT_VOXELS:
                    keep |= (labeled == lbl)
            # combine with the largest component mask to be strict
            shell = keep & largest
        else:
            shell = largest

    # Pre-closing dilation (thicken shell slightly)
    if DILATE_RINGS > 0:
        structure = np.zeros((3,3,3), dtype=bool)
        structure[1,1,0] = structure[1,1,2] = True
        structure[1,0,1] = structure[1,2,1] = True
        structure[0,1,1] = structure[2,1,1] = True
        shell = binary_dilation(shell, structure=structure, iterations=int(DILATE_RINGS))

    # Morphological closing to seal pinholes (dilate then erode)
    # Morphological closing with larger kernel
    if CLOSING_ITERS > 0:
        k = 5  # try 5; set 7 if still leaky (slower)
        structure = np.ones((k, k, k), dtype=bool)
        for _ in range(int(CLOSING_ITERS)):
            shell = binary_dilation(shell, structure=structure, iterations=1)
            shell = binary_erosion(shell, structure=structure, iterations=1)

    # shell-only volume (lower-ish bound)
    shell_voxels = int(shell.sum())
    vol_shell = shell_voxels * (VOXEL_SIZE ** 3)

    # Flood fill exterior air; interior = not exterior
    exterior = flood_fill_exterior(shell)
    interior = ~exterior  # includes the shell too
    # For a clean solid volume, take interior (everything not connected to outside)
    solid_voxels = int(interior.sum())
    vol_solid = solid_voxels * (VOXEL_SIZE ** 3)

    report = {
        "input_object_pcd": str(p_in),
        "voxel_size_m": VOXEL_SIZE,
        "padding_voxels": PADDING_VOXELS,
        "dilate_rings": DILATE_RINGS,
        "closing_iters": CLOSING_ITERS,
        "remove_small_components": REMOVE_SMALL_COMPONENTS,
        "min_component_voxels": MIN_COMPONENT_VOXELS,
        "grid_dims_zyx": [Z, Y, X],
        "shell": {
            "occupied_voxels": shell_voxels,
            "volume_m3": vol_shell
        },
        "solid_fill": {
            "solid_voxels": solid_voxels,
            "volume_m3": vol_solid
        }
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / "voxel_solid_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote report: {out_json}")


if __name__ == "__main__":
    main()


import csv
import json
from pathlib import Path
from typing import Optional, Iterable

import numpy as np

try:
    import open3d as o3d
except Exception as e:
    raise SystemExit("open3d is required. Install with: pip install open3d") from e

try:
    import trimesh
except Exception as e:
    raise SystemExit("trimesh is required. Install with: pip install trimesh") from e

try:
    from scipy.spatial import Delaunay
except Exception as e:
    raise SystemExit("scipy is required (Delaunay). Install with: pip install scipy") from e

# ---- USER CONFIG ----
INPUT_OBJECT_PCD: Optional[str] = None  # e.g., r"C:\...\outputs_step3\step3_object_cluster.pcd"
TARGET_SAMPLE_POINTS = 10000            # up to this many points
ALPHAS = [0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.18, 0.22, 0.26]  # meters
OUT_DIR = Path.cwd() / "outputs_alpha_shape_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_latest_object_pcd() -> Optional[Path]:
    base = Path.cwd() / "outputs_step3"
    cands = sorted(base.glob("step3_object_cluster.pcd"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

def random_downsample(points: np.ndarray, n: int) -> np.ndarray:
    if points.shape[0] <= n:
        return points
    idx = np.random.choice(points.shape[0], size=n, replace=False)
    return points[idx]

def circumradius_tetra(p):
    a, b, c, d = p
    A = np.vstack([b - a, c - a, d - a]).T  # 3x3
    detA = np.linalg.det(A)
    if abs(detA) < 1e-12:
        return np.inf
    asq, bsq, csq, dsq = (np.dot(a,a), np.dot(b,b), np.dot(c,c), np.dot(d,d))
    rhs = 0.5 * np.array([bsq - asq, csq - asq, dsq - asq])
    x = np.linalg.solve(A, rhs)
    center = a + x
    R = np.linalg.norm(a - center)
    return R

def tetra_volume(p):
    a, b, c, d = p
    return abs(np.dot((b - a), np.cross((c - a), (d - a)))) / 6.0

def alpha_shape_3d_volume(points: np.ndarray, alpha: float):
    dela = Delaunay(points)
    simplices = dela.simplices  # (M,4)
    pts = points

    keep = []
    for tet in simplices:
        p = pts[tet]
        R = circumradius_tetra(p)
        if np.isfinite(R) and R <= alpha:
            keep.append(tet)

    if not keep:
        return 0.0, None

    from collections import defaultdict
    tri_count = defaultdict(int)
    def faces_of_tet(t):
        a,b,c,d = t
        return [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
    for tet in keep:
        for tri in faces_of_tet(tet):
            tri = tuple(sorted(tri))
            tri_count[tri] += 1

    boundary_tris = [tri for tri, cnt in tri_count.items() if cnt == 1]
    if len(boundary_tris) == 0:
        return 0.0, None

    vertices = points
    faces = np.array(boundary_tris, dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    if not mesh.is_watertight:
        mesh = mesh.fill_holes()
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()

    vol = float(mesh.volume) if mesh.is_volume or mesh.is_watertight else float(mesh.convex_hull.volume)
    return vol, mesh

def main():
    p_in = Path(INPUT_OBJECT_PCD) if INPUT_OBJECT_PCD else find_latest_object_pcd()
    if p_in is None or not p_in.exists():
        raise SystemExit("Could not find object PCD. Provide INPUT_OBJECT_PCD or run Step 3 first.")
    print(f"Object PCD: {p_in}")

    pcd = o3d.io.read_point_cloud(str(p_in))
    pts = np.asarray(pcd.points, dtype=np.float64)
    pts_ds = random_downsample(pts, TARGET_SAMPLE_POINTS)
    print(f"Points used: {pts_ds.shape[0]}")

    csv_path = OUT_DIR / "alpha_shape_sweep.csv"
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["alpha_m", "volume_m3", "mesh_saved"])
        for alpha in ALPHAS:
            print(f"Trying alpha = {alpha} m ...")
            try:
                vol, mesh = alpha_shape_3d_volume(pts_ds, alpha)
                saved = ""
                if mesh is not None and vol > 0:
                    out_mesh = OUT_DIR / f"alpha_{alpha:.3f}.ply"
                    mesh.export(out_mesh)
                    saved = str(out_mesh)
                writer.writerow([alpha, vol, saved])
                print(f"  volume = {vol:.6f} m^3  saved={bool(saved)}")
            except Exception as e:
                writer.writerow([alpha, "error", str(e)])
                print(f"  error: {e}")

    print(f"Wrote CSV: {csv_path}")

if __name__ == "__main__":
    main()

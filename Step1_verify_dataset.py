
import os
import sys
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Optional dependencies
try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

# open3d is great for PCDs if available
try:
    import open3d as o3d
except Exception:
    o3d = None

def human_size(num_bytes: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"

def read_text_head(p: Path, n_lines: int = 20) -> str:
    try:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            lines = [next(f) for _ in range(n_lines)]
        return ''.join(lines)
    except StopIteration:
        return ''.join(lines)
    except Exception as e:
        return f"[ERROR reading text: {e}]"

def inspect_png(p: Path) -> Dict[str, Any]:
    info = {"path": str(p), "type": "PNG"}
    # Try cv2 first
    if cv2 is not None:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            shape = img.shape
            dtype = str(img.dtype)
            info.update({"height": shape[0], "width": shape[1], "channels": shape[2] if len(shape)==3 else 1, "dtype": dtype})
            return info
    # Fallback to PIL
    if Image is not None:
        try:
            with Image.open(p) as im:
                info.update({"width": im.width, "height": im.height, "mode": im.mode})
        except Exception as e:
            info["error"] = f"PIL open failed: {e}"
    else:
        info["warning"] = "Neither OpenCV nor PIL available to inspect image."
    return info

def inspect_pcd(p: Path) -> Dict[str, Any]:
    info = {"path": str(p), "type": "PCD"}
    # Try open3d
    if o3d is not None:
        try:
            pcd = o3d.io.read_point_cloud(str(p))
            # open3d returns empty pcd if failed sometimes; check points
            n_pts = 0 if pcd.is_empty() else len(pcd.points)
            has_colors = (not pcd.is_empty()) and len(pcd.colors) == len(pcd.points) and len(pcd.points) > 0
            info.update({"points": n_pts, "has_colors": bool(has_colors)})
            return info
        except Exception as e:
            info["open3d_error"] = f"{e}"
    # Fallback: read PCD header manually
    try:
        with p.open('rb') as f:
            head = f.read(4096).decode('utf-8', errors='ignore')
        info["header_head"] = '\n'.join(head.splitlines()[:20])
        # crude parse
        for line in head.splitlines():
            if line.upper().startswith("POINTS"):
                try:
                    info["points_declared"] = int(line.split()[1])
                except Exception:
                    pass
            if line.upper().startswith("FIELDS"):
                info["fields"] = line.split()[1:]
            if line.upper().startswith("DATA"):
                info["data"] = line.split()[1] if len(line.split())>1 else "unknown"
    except Exception as e:
        info["error"] = f"[ERROR reading PCD header: {e}]"
    return info

def inspect_bag(p: Path) -> Dict[str, Any]:
    info = {"path": str(p), "type": "BAG", "size": human_size(p.stat().st_size)}
    # We won't attempt to parse without ROS tools; just report existence & size.
    return info

def summarize_dir(dir_path: Path, expected_exts: Tuple[str, ...]) -> Dict[str, Any]:
    summary = {
        "directory": str(dir_path),
        "exists": dir_path.exists(),
        "file_count": 0,
        "extensions_count": {},
        "samples": []
    }
    if not dir_path.exists():
        return summary
    files = [p for p in dir_path.iterdir() if p.is_file()]
    summary["file_count"] = len(files)
    # Count extensions
    exts: Dict[str, int] = {}
    for p in files:
        ext = p.suffix.lower()
        exts[ext] = exts.get(ext, 0) + 1
    summary["extensions_count"] = exts
    # Pick up to 3 samples matching expected_exts (or any if empty)
    chosen = []
    if expected_exts:
        for ext in expected_exts:
            chosen.extend([p for p in files if p.suffix.lower() == ext][:1])
    if not chosen:
        chosen = files[:3]
    # Inspect samples
    inspections = []
    for p in chosen[:3]:
        if p.suffix.lower() in ('.png',):
            inspections.append(inspect_png(p))
        elif p.suffix.lower() in ('.pcd',):
            inspections.append(inspect_pcd(p))
        elif p.suffix.lower() in ('.txt', '.yaml', '.yml'):
            head = read_text_head(p, n_lines=20)
            inspections.append({"path": str(p), "type": "TEXT", "head": head})
        elif p.suffix.lower() in ('.bag', '.db3'):
            inspections.append(inspect_bag(p))
        else:
            inspections.append({"path": str(p), "type": f"Unknown ({p.suffix})", "size": human_size(p.stat().st_size)})
    summary["samples"] = inspections
    return summary

def main():
    # Update these paths as needed (Windows raw strings helpful for backslashes)
    paths = [
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_color_camera_info", ('.txt', '.yaml', '.yml')),
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_color_image_raw", ('.png',)),
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_depth_camera_info", ('.txt', '.yaml', '.yml')),
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_depth_image_raw", ('.png',)),
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_depth_points", ('.pcd',)),
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_left_ir_image_raw", ('.png',)),
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\camera_right_ir_image_raw", ('.png',)),
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\raw\test\depth_registered_colored_pointclouds", ('.pcd',)),
        (r"C:\Users\Samous\Desktop\Project_2\Project_2\rosbags", ('.bag', '.db3'))
    ]

    all_summaries: List[Dict[str, Any]] = []
    for path_str, exts in paths:
        d = Path(path_str)
        summary = summarize_dir(d, exts)
        all_summaries.append(summary)

    # Pretty print to console
    print("="*80)
    print("DATASET INVENTORY SUMMARY")
    print("="*80)
    for s in all_summaries:
        print(f"\nDirectory: {s['directory']}")
        print(f"  Exists: {s['exists']}")
        print(f"  File count: {s['file_count']}")
        print(f"  Extensions: {s['extensions_count']}")
        if s['samples']:
            print("  Sample inspections:")
            for item in s['samples']:
                # Limit long heads
                if 'head' in item and isinstance(item['head'], str) and len(item['head']) > 400:
                    item = dict(item)  # shallow copy
                    item['head'] = item['head'][:400] + ' ... [truncated]'
                print("   -", json.dumps(item, ensure_ascii=False))

    # Also write a CSV summary of counts
    csv_path = Path.cwd() / "dataset_inventory.csv"
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["directory", "exists", "file_count", "extensions_json"])
        for s in all_summaries:
            writer.writerow([s['directory'], s['exists'], s['file_count'], json.dumps(s['extensions_count'])])

    print(f"\nCSV summary written to: {csv_path}")
    print("Tip: If you see 'exists: False' for any directory, double-check the path or drive letter.")
    print("\nOptional dependencies:")
    print(" - OpenCV (cv2) for image inspection")
    print(" - Pillow (PIL) for image inspection fallback")
    print(" - open3d for PCD parsing")
    print("Install with: pip install opencv-python pillow open3d")

if __name__ == "__main__":
    main()

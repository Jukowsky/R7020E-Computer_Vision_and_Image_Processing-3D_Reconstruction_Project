import numpy as np
import cv2
import os
import yaml

def compute_disparity(left_image, right_image):
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    #Create stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=96,  # Try: 64, 96, 128, 160 (must be divisible by 16)
        blockSize=11,  # Try: 5, 7, 9, 11 (odd numbers only)
        P1=8 * 3 * 7 ** 2,  # Smoothness penalty 1
        P2=32 * 3 * 7 ** 2,  # Smoothness penalty 2 (larger = smoother)
        disp12MaxDiff=1,
        uniquenessRatio=10,  # Try: 5-15 (higher = more filtering)
        speckleWindowSize=100,  # Remove noise spots
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(left_gray, right_gray)

    disparity_normalized = cv2.normalize(disparity,None,0,255,cv2.NORM_MINMAX)
    disparity_normalized = np.uint8(disparity_normalized)

    return disparity, disparity_normalized

def show_results(left_image, disparity_normalized):
    """Display original and disparity side by side"""
    # Resize disparity to match if needed
    if len(left_img.shape) == 3:
        disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    else:
        disparity_color = cv2.cvtColor(disparity_normalized, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([left_img, disparity_color])
    cv2.imshow('Left Image | Disparity Map', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def disparity_to_pointcloud(disparity, K, baseline=0.05):
    """
    Convert Disparity Map to PointCloud
    disparity: disparity map from 2nd Phase
    K: camera intrinsic matrix
    baseline: distance between cameras (0.05m = 5cm Default)
    """
    # Get focal length and center from K matrix:

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Create mesh grid of pixel coordinates
    h, w= disparity.shape[:2] # where disparity shape is (400, 640)
    x = np.arange(0, w)
    y = np.arange(0, h)
    xx, yy = np.meshgrid(x, y)

    # Avoid division by zero
    disparity = disparity.astype(np.float32) / 16.0 # OpenCV disparity is scaled by 16
    disparity[disparity == 0] = 0.1 #Small value to avoid inf

    # Calculate depth (Z)
    Z = (fx * baseline) / disparity

    # Calculate X and Y
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy

    # Stack to create point cloud (Nx3)
    points = np.stack([X, Y, Z], axis=-1)

    # Reshape to (H*W, 3)
    points = points.reshape(-1,3)

    print(f"Generated {len(points)} 3D points")
    return points

def filter_pointcloud(points, max_depth = 5.0):
    """
    Remove points that are too far or invalid
    """
    mask = (points[:, 2] > 0) & (points[:, 2] < max_depth)
    filtered = points[mask]

    print(f"Filtered {len(filtered)} valid points (removed {len(points) - len(filtered)} points)")
    return filtered

def save_pointcloud(points, filename="pointcloud.ply"):
    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write points
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"Saved {filename}")

def segment_object(points):
    """
    Remove floor and walls, keep all objects
    """
    # Step 1: Remove floor (lowest Y points)
    floor_height = np.percentile(points[:, 1], 5)  # Bottom 5%
    points = points[points[:, 1] > floor_height + 0.1]  # Remove floor

    # Step 2: Remove walls (far depth)
    points = points[points[:, 2] < 2.5]  # Keep objects closer than 2.5m

    # Step 3: Remove ceiling (highest Y points)
    ceiling_height = np.percentile(points[:, 1], 95)  # Top 5%
    points = points[points[:, 1] < ceiling_height - 0.1]

    print(f"After removing floor/walls/ceiling: {len(points)} points")
    return points

def remove_outliers(points, max_distance=0.1):
    """
    Remove scattered outlier points
    Keeps points that have neighbors nearby
    Returns cleaned (wo outliers)
    """
    from scipy.spatial import cKDTree

    # Build tree for fast neighbor search
    tree = cKDTree(points)

    # Count neighbors within distance
    neighbor_counts = []
    for point in points:
        neighbors = tree.query_ball_point(point, max_distance)
        neighbor_counts.append(len(neighbors))

    # Keep points with enough neighbors
    neighbor_counts = np.array(neighbor_counts)
    threshold = np.percentile(neighbor_counts, 20) # Keeping top 80%
    mask = neighbor_counts >= threshold

    cleaned = points[mask]

    print(f"Outlier removal:")
    print(f" Before: {len(points)}")
    print(f" After: {len(cleaned)}")

    return cleaned


def save_pointcloud_with_color(points, filename="segmented_object.ply", color=[255, 0, 0]):
    """Save colored point cloud"""

    with open(filename, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Write points with color
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {color[0]} {color[1]} {color[2]}\n")

    print(f"Saved {filename}")


def calculate_volume_convex_hull(points):
    """
    Calculate volume using convex hull
    """
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(points)
        volume = hull.volume

        print(f"Convex Hull Volume: {volume:.6f} cubic meters")
        print(f"Convex Hull Volume: {volume*1000:.2f} liters")

        return volume
    except Exception as e:
        print(f"Error calculating convex hull volume: {e}")
        return None


def calculate_volume_voxel(points, voxel_size=0.01):
    """
    Calculate volume by counting voxels - SIMPLE & ROBUST

    points: 3D point cloud (N x 3)
    voxel_size: size of each voxel cube (default 1cm = 0.01m)
    """

    # Round points to voxel grid
    voxels = np.round(points / voxel_size).astype(int)

    # Get unique voxels (remove duplicates)
    unique_voxels = np.unique(voxels, axis=0)

    # Calculate volume
    num_voxels = len(unique_voxels)
    voxel_volume = voxel_size ** 3
    total_volume = num_voxels * voxel_volume

    print(f"Voxel-based Volume:")
    print(f"  Voxel size: {voxel_size * 100:.1f} cm")
    print(f"  Number of voxels: {num_voxels}")
    print(f"  Volume: {total_volume:.6f} cubic meters")
    print(f"  Volume: {total_volume * 1000:.2f} liters")

    return total_volume


def calculate_volume_bounding_box(points):
    """
    Calculate bounding box volume - UPPER BOUND
    This gives maximum possible volume
    """

    # Get min and max in each dimension
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Calculate dimensions
    dimensions = max_coords - min_coords

    # Calculate volume
    volume = dimensions[0] * dimensions[1] * dimensions[2]

    print(f"Bounding Box Volume:")
    print(f"  Dimensions (X, Y, Z): {dimensions[0]:.3f} x {dimensions[1]:.3f} x {dimensions[2]:.3f} meters")
    print(f"  Volume: {volume:.6f} cubic meters")
    print(f"  Volume: {volume * 1000:.2f} liters")

    return volume


if __name__ == "__main__":
    """
    Timestamps are not fitting to each other in right and left files. Therefore this approach has been commented out.
    # Load your stereo images
    timestamp = '1727164782244389392'
    left_img = cv2.imread(f"raw/test/camera_left_ir_image_raw/left_ir_image_{timestamp}.png")
    right_img = cv2.imread(f"raw/test/camera_right_ir_image_raw/right_ir_image_{timestamp}.png")
    """

    data_folder = "raw/test"
    file_index = 1

    left_folder = os.path.join(data_folder, "camera_left_ir_image_raw")
    right_folder = os.path.join(data_folder, "camera_right_ir_image_raw")

    left_files = sorted(os.listdir(left_folder))
    right_files = sorted(os.listdir(right_folder))

    left_path = os.path.join(left_folder, left_files[file_index])
    right_path = os.path.join(right_folder, right_files[file_index])

    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    print("Computing disparity map...")
    disparity, disparity_vis = compute_disparity(left_img, right_img)

    print(f"Disparity shape: {disparity.shape}")
    print(f"Disparity range: {disparity.min()} to {disparity.max()}")

    # Show results
    show_results(left_img, disparity_vis)

    # Save disparity map
    cv2.imwrite(f"disparity_map_{file_index}.png", disparity_vis)
    print(f"Saved disparity_map_{file_index}.png")

    # Load camera matrix from the camera_info files. Assuming K matrix are same for all files.
    K = np.array(
        [[306.000244140625, 0.0, 318.4753112792969],
         [0.0, 306.1123352050781, 201.36949157714844],
         [0.0, 0.0, 1.0]]
    )

    print("Converting disparity to 3D point cloud...")
    points = disparity_to_pointcloud(disparity, K, baseline=0.05)

    print("Filtering point cloud...")
    points_filtered = filter_pointcloud(points, max_depth=3.0)

    print("Saving point cloud...")
    save_pointcloud(points_filtered, "pointcloud.ply")

    print("=== Segmentation ===\n")

    # Step 1: Segment by depth
    print("Step 1: Depth-based segmentation...")
    segmented = segment_object(points)

    # Step 2: Remove outliers (optional - comment out if too slow)
    print("\nStep 2: Removing outliers...")
    try:
        segmented = remove_outliers(segmented, max_distance=0.05)
    except:
        print("  Skipping outlier removal (scipy not available)")

    # Step 3: Save segmented object
    print("\nStep 3: Saving segmented object...")
    save_pointcloud_with_color(segmented, "segmented_object.ply", color=[0, 255, 0])

    print("\n=== Segmentation Completed ===")

    print("=== Volume Estimation ===")

    print("\nMethod 1: Voxel-based")
    volume_voxel = calculate_volume_voxel(segmented, voxel_size=0.01)

    print("\nMethod 2: Convex Hull")
    volume_hull = calculate_volume_convex_hull(segmented)

    print("\nMethod 3: Bounding Box")
    volume_bbox = calculate_volume_bounding_box(segmented)







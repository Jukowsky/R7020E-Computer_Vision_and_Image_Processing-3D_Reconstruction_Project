import cv2
import numpy as np
import os

# Simple script to explore your stereo dataset

# Path to your data (update this to your actual path)
DATA_PATH = "raw/test"

def load_images(folder, num=5):
    """Load first few images from a folder"""
    files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])[:num]
    images = []
    for f in files:
        img = cv2.imread(os.path.join(folder, f))
        images.append(img)
    return images, files

def show_stereo_pair(left_img, right_img):
    """Display left and right images side by side"""
    combined = np.hstack([left_img, right_img])
    cv2.imshow('Left | Right', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main exploration
print("=== Phase 1: Exploring Dataset ===\n")

# 1. Load left images
left_folder = os.path.join(DATA_PATH, "camera_left_ir_image_raw")
print(f"Loading left images from: {left_folder}")
left_images, left_names = load_images(left_folder)
print(f"Found {len(left_images)} left images")
print(f"Image size: {left_images[0].shape}\n")

# 2. Load right images
right_folder = os.path.join(DATA_PATH, "camera_right_ir_image_raw")
print(f"Loading right images from: {right_folder}")
right_images, right_names = load_images(right_folder)
print(f"Found {len(right_images)} right images\n")

# 3. Display first stereo pair
print("Displaying first stereo pair...")
show_stereo_pair(left_images[0], right_images[0])

# 4. Check if depth data exists
depth_folder = os.path.join(DATA_PATH, "camera_depth_image_raw")
if os.path.exists(depth_folder):
    print("\n✓ Depth data available!")
    depth_files = os.listdir(depth_folder)
    print(f"Found {len(depth_files)} depth images")
else:
    print("\n✗ No depth data found")

print("\n=== Phase 1 Complete ===")
print("Next: Phase 2 - Create disparity map")
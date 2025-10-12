import numpy as np
import cv2
import os

def compute_disparity(left_image, right_image):
    left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    #Create stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
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

    # Example usage


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
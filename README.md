# Project #2: 3D Reconstruction and Volume Estimation

**Course:** R7020E Computer Vision  
**Team Size:** 4 students per group  
**Project Duration:** 4 weeks  
**Target Object:** Tank/Container (~60-65 Liters)

---

## 📋 Project Overview

This project focuses on building a complete pipeline to reconstruct objects in 3D from stereo camera data, segment them from the background, and accurately estimate their volume. The system processes stereo vision data to create 3D point clouds, applies segmentation techniques to isolate target objects, and implements multiple volume estimation algorithms.

---

## 🎯 Objectives

1. **3D Reconstruction**: Develop a stereo vision pipeline using OpenCV or SLAM-based methods to generate 3D point clouds or meshes
2. **Object Segmentation**: Segment target objects using semantic segmentation, edge detection, or instance segmentation techniques
3. **Volume Estimation**: Calculate object volume from the segmented 3D model with ±30% precision

---

## 📊 Dataset Structure

The project uses custom recordings provided in ROS bag format, organized as follows:

```
Project_2/
├── raw/
│   └── test/
│       ├── camera_color_camera_info/      # RGB camera calibration data (149 items)
│       ├── camera_color_image_raw/        # RGB images (149 items)
│       ├── camera_depth_camera_info/      # Depth camera calibration data (83 items)
│       ├── camera_depth_image_raw/        # Raw depth images (83 items)
│       ├── camera_depth_points/           # 3D point data (102 items)
│       ├── camera_left_ir_image_raw/      # Left infrared stereo image (121 items)
│       ├── camera_right_ir_image_raw/     # Right infrared stereo image (124 items)
│       └── depth_registered_colored_pointclouds/  # Preprocessed colored point clouds (15 items)
└── rosbags/
    └── project2_test_2024-09-24-09-59-41.bag  # Original ROS bag file (690 MB)
```

---

## 🏗️ System Architecture

### Pipeline Overview

```
Input Data (Stereo Images / Depth Maps)
           ↓
    Calibration & Preprocessing
           ↓
    3D Reconstruction (Point Cloud Generation)
           ↓
    Segmentation (Object Isolation)
           ↓
    Volume Estimation
           ↓
    Evaluation & Validation
```

### Key Components

1. **Data Loading Module**
   - ROS bag parser
   - Camera calibration loader
   - Point cloud reader

2. **3D Reconstruction Module**
   - Stereo matching (if using raw stereo pairs)
   - Disparity to depth conversion
   - Point cloud generation
   - Coordinate system transformation

3. **Segmentation Module**
   - Ground plane removal (RANSAC)
   - Clustering (DBSCAN/Euclidean)
   - Color-based refinement
   - Noise filtering

4. **Volume Estimation Module**
   - Convex Hull method
   - Voxel-based estimation
   - Alpha Shapes method
   - Mesh reconstruction (Poisson/Ball-pivoting)

5. **Evaluation Module**
   - Segmentation accuracy metrics
   - Volume error calculation
   - Visualization tools

---

## 🔧 Technical Requirements

### Software Dependencies

```python
# Core libraries
opencv-python>=4.8.0
numpy>=1.24.0
open3d>=0.17.0

# ROS bag handling
rosbag>=1.15.0
sensor_msgs>=1.13.0

# Scientific computing
scipy>=1.11.0
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
```

### Hardware Requirements

- **Minimum**: 8GB RAM, 4-core CPU
- **Recommended**: 16GB RAM, GPU support for acceleration
- **Storage**: ~1GB for data and intermediate results

---

## 📈 Performance Metrics

### Target Specifications

| Metric | Target Value | Notes |
|--------|--------------|-------|
| **Volume Estimation Precision** | ±30% of ground truth | Covers end-to-end pipeline accuracy |
| **Ground Truth Volume** | 60-65 Liters | Tank only (excluding appendages) |
| **Segmentation Accuracy** | IoU > 0.80 | Intersection over Union with manual annotation |
| **Processing Time** | < 30 seconds per frame | For real-time feasibility assessment |

### Evaluation Criteria

1. **Segmentation Quality**
   - Visual inspection
   - IoU calculation (if ground truth available)
   - False positive/negative rate

2. **Volume Accuracy**
   - Absolute error: |V_estimated - V_ground_truth|
   - Relative error: (|V_estimated - V_ground_truth| / V_ground_truth) × 100%
   - Consistency across multiple frames

3. **Robustness**
   - Performance under different lighting conditions
   - Handling of occlusions
   - Noise resilience

---

## 🚀 Implementation Approach

### Phase 1: Data Exploration (Week 1)
- [ ] Load and visualize ROS bag data
- [ ] Explore stereo images and depth maps
- [ ] Understand camera calibration parameters
- [ ] Visualize point clouds using Open3D
- [ ] Identify data quality issues

### Phase 2: 3D Reconstruction (Week 2)
- [ ] Implement stereo matching (if needed)
- [ ] Generate point clouds from depth data
- [ ] Apply coordinate transformations
- [ ] Filter outliers and noise
- [ ] Validate reconstruction quality

### Phase 3: Segmentation (Week 2-3)
- [ ] Implement ground plane removal
- [ ] Apply clustering algorithms
- [ ] Refine segmentation with color data
- [ ] Optimize parameters
- [ ] Validate segmentation results

### Phase 4: Volume Estimation (Week 3)
- [ ] Implement Convex Hull method
- [ ] Implement Voxel-based method
- [ ] Implement Alpha Shapes method
- [ ] Implement Mesh reconstruction method
- [ ] Compare all methods against ground truth

### Phase 5: Evaluation & Documentation (Week 4)
- [ ] Run comprehensive experiments
- [ ] Generate performance metrics
- [ ] Create visualizations
- [ ] Document challenges and limitations
- [ ] Write final report

---

## 📝 Deliverables

### Code Deliverables
1. Complete pipeline implementation
2. Modular, well-documented code
3. Configuration files for parameters
4. Unit tests for key components

### Report Sections

#### 1. System Architecture & Design
- Stereo vision pipeline description
- Segmentation method rationale
- Volume estimation mechanism
- Algorithm selection justification

#### 2. Implementation Details
- Data preprocessing steps
- Algorithm parameters and tuning
- Coordinate system transformations
- Error handling strategies

#### 3. Performance Metrics
- Segmentation accuracy results
- Volume estimation precision
- Comparison of different methods
- Processing time analysis

#### 4. Challenges and Limitations
- Depth estimation challenges
  - Stereo matching errors
  - Sensor noise and artifacts
  - Limited field of view
  
- Segmentation challenges
  - Background complexity
  - Object texture and reflectance
  - Occlusion handling
  
- Volume estimation limitations
  - Point cloud density
  - Surface reconstruction accuracy
  - Computational complexity

#### 5. Results and Discussion
- Visual results (point clouds, segmentations, meshes)
- Quantitative analysis
- Method comparison
- Future improvements

---

## 🛠️ Recommended Tools & Libraries

### Core Libraries
- **Open3D**: 3D data processing and visualization
- **OpenCV**: Image processing and stereo vision
- **NumPy/SciPy**: Numerical computations
- **scikit-learn**: Clustering algorithms

### Visualization
- **Open3D Visualizer**: Interactive 3D viewer
- **Matplotlib**: 2D plots and charts
- **Plotly**: Interactive 3D plots (optional)

### ROS Integration
- **rosbag**: Read ROS bag files
- **cv_bridge**: Convert ROS images to OpenCV format
- **sensor_msgs**: Handle point cloud messages

---

## 📚 Key Algorithms

### 1. Stereo Reconstruction
- **Stereo Block Matching**: Dense disparity estimation
- **Semi-Global Matching (SGM)**: Better quality at higher cost
- **Depth map filtering**: Median filter, bilateral filter

### 2. Segmentation
- **RANSAC Plane Fitting**: Ground plane removal
- **DBSCAN Clustering**: Density-based object clustering
- **Euclidean Clustering**: Distance-based grouping
- **Statistical Outlier Removal**: Noise filtering

### 3. Volume Estimation
- **Convex Hull**: Fast but overestimates for concave objects
- **Voxel Grid**: Adjustable precision, good balance
- **Alpha Shapes**: Better than convex hull for concave objects
- **Poisson Reconstruction**: High-quality mesh, accurate volume

---

## 🎓 Learning Outcomes

By completing this project, you will:

1. Understand stereo vision principles and 3D reconstruction
2. Master point cloud processing techniques
3. Implement and compare segmentation algorithms
4. Evaluate volume estimation methods
5. Develop skills in scientific reporting and analysis
6. Gain experience with industry-standard computer vision tools

---

## 📖 References

### Academic Papers
- Zhang, Z. (2000). "A flexible new technique for camera calibration"
- Fischler, M. A., & Bolles, R. C. (1981). "Random sample consensus: a paradigm for model fitting"
- Edelsbrunner, H., & Mücke, E. P. (1994). "Three-dimensional alpha shapes"

### Documentation
- [Open3D Documentation](http://www.open3d.org/docs/)
- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html)
- [ROS bag Python API](http://wiki.ros.org/rosbag/Code%20API)

### Tutorials
- Open3D Point Cloud Processing Tutorial
- Stereo Vision and Depth Estimation
- 3D Reconstruction from Multiple Views

---

## 👥 Team Collaboration

### Recommended Task Distribution

**Team Member 1**: Data Loading & Preprocessing
- ROS bag parsing
- Camera calibration
- Initial point cloud generation

**Team Member 2**: 3D Reconstruction
- Stereo matching implementation
- Point cloud refinement
- Coordinate transformations

**Team Member 3**: Segmentation
- Ground plane removal
- Clustering algorithms
- Segmentation refinement

**Team Member 4**: Volume Estimation & Evaluation
- Multiple volume methods
- Metrics calculation
- Visualization and reporting

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Point cloud appears noisy or sparse
- **Solution**: Adjust stereo matching parameters, apply statistical outlier removal

**Issue**: Segmentation includes background objects
- **Solution**: Tune RANSAC parameters, adjust clustering distance threshold

**Issue**: Volume estimation highly inaccurate
- **Solution**: Check point cloud units, verify segmentation quality, try different methods

**Issue**: ROS bag reading errors
- **Solution**: Ensure correct ROS version, check Python environment compatibility

---

## 📧 Contact & Support

For questions and discussions:
- Team communication: [Setup team Slack/Discord]
- Course instructor: [Instructor contact]
- Office hours: [Schedule]

---

## 📄 License

This project is for educational purposes as part of the R7020E Computer Vision course.

---

**Last Updated**: October 2025  
**Version**: 1.0  
**Status**: In Progress

---

## ✅ Quick Start Checklist

- [ ] Clone/download the project data
- [ ] Set up Python environment with required libraries
- [ ] Verify ROS bag can be read
- [ ] Run initial point cloud visualization
- [ ] Review project requirements and metrics
- [ ] Plan team task distribution
- [ ] Set up version control (Git)
- [ ] Create project timeline

---

**Good luck with your 3D reconstruction project! 🚀**

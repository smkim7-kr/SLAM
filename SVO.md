
# SLAM and SVO: Overview and Analysis

## Introduction to SLAM
SLAM (Simultaneous Localization and Mapping) is a method used in autonomous robotics to navigate unknown environments. It enables robots to map their surroundings while tracking their own location.

## Introduction to SVO (Semi-Direct Visual Odometry)
SVO, or Semi-Direct Visual Odometry, was first introduced in 2014 by Forster et al. It combines feature-based and direct methods to optimize both the visual odometry (VO) and structure simultaneously. SVO is specifically designed for lightweight embedded systems, making it ideal for applications like drones or mobile robots.

### Key Contributors
- **윤성호**: Researcher at KAIST, with past experience in LG Electronics, focused on SLAM and machine learning.
- **Presentation Date**: February 23, 2020
- **Transcript Link**: [SLAM DUNK 2020 | 윤성호, 이재민 발표 - YouTube](https://www.youtube.com/watch?v=1jWCFDcisWM)

---

## SVO Methodology

### Characteristics of SVO
- **Hybrid Approach**: Combines feature-based and direct methods.
- **Efficient Processing**: Approximately 2.5 ms per frame, which is suitable for real-time applications.
- **Application**: Effective for embedded systems, with adaptability to multi-camera setups, fisheye lenses, and more.

### Direct vs. Feature-Based Methods in SVO
- **Direct Methods**: Focuses on minimizing photometric errors between pixel intensities. It works well even in environments where traditional feature extraction is challenging.
- **Feature-Based Methods**: Minimizes reprojection errors, making it robust in handling loop closures and wide-baseline matching.

#### Strengths and Weaknesses
1. **Direct Methods**:
   - **Pros**: Accurate in low-feature environments.
   - **Cons**: Difficulty in obtaining accurate covariance, less integration with inertial sensors.
2. **Feature-Based Methods**:
   - **Pros**: Strong for wide-baseline matching and feature-rich scenes.
   - **Cons**: Slow processing due to extraction and matching.

### Key Components of SVO System
The SVO system is divided into **Motion Estimation** and **Mapping** threads.

#### Motion Estimation Thread
1. **Sparse Image Alignment**: Aligns the current frame with the previous one, calculating relative poses.
2. **Relaxation**: Aligns frames to keyframes instead of only the previous frame, reducing drift.
3. **Refinement**: Uses local bundle adjustment for optimization.

#### Mapping Thread
- **Probabilistic Depth Filtering**: Initializes and refines depth filters using a recursive Bayesian update. Depth values are continually adjusted until they converge.

### Mathematical Basis in SVO
1. **Transformation Representations**: Represented in the Lie algebra se(3) space, with SE(3) denoting the space of 3D transformations. Twist coordinates map to SE(3) through an exponential map.
2. **Optimization**: SVO employs least squares methods, typically through iterative Gauss-Newton optimization, to derive the relative pose \( T_{k, k-1} \).
3. **Inverse Compositional Algorithm**: Allows pre-computation of the Jacobian for speed, aligning camera poses across multiple frames.

### Detailed Steps in SVO

#### A. Sparse Image Alignment
Aligns frames to compute initial relative poses. This is followed by frame-to-frame adjustments using the inverse compositional Lucas-Kanade algorithm.

#### B. Relaxation Step
Optimizes the pose alignment to keyframes instead of only the previous frame, which reduces accumulated drift. This step violates epipolar constraints but improves accuracy.

#### C. Refinement Step
Refines both the camera pose and 3D points by minimizing reprojection errors, resulting in a locally optimized bundle adjustment.

---

## Depth Estimation and Mapping

SVO integrates a depth-filtering approach for continuous 2D-to-3D mapping. This depth filtering is accomplished by:
- **Patch-Based Matching**: Searches along the epipolar line for corresponding points to estimate depth.
- **Gaussian-Uniform Mixture Model**: Models depth uncertainty, distinguishing between inliers and outliers.

### Patch-Based Feature Extraction
Uses FAST corners for feature points, distributing these points evenly across cells in an image grid. The depth filter is initialized with high uncertainty and refined through multiple observations.

### Handling Outliers
Outliers are managed by introducing a Gaussian-Uniform mixture model for the depth estimate, which gradually refines as more observations are collected. The system employs an inverse depth representation for large scenes, enhancing stability over long distances.

---

## Applications and Extensions

SVO has been successfully adapted to:
1. **Embedded Systems**: Particularly useful in drones where computational resources are limited.
2. **Wide Field of View Cameras**: SVO can handle fisheye lenses and other wide FoV setups by tracking edges and using multiple cameras.
3. **CNN-SVO**: An enhanced version introduced at ICRA 2019, which integrates CNNs to improve accuracy and robustness.

---

## Limitations and Improvements
- **Drift Accumulation**: Although SVO includes drift mitigation, longer sequences may still experience drift.
- **Reliance on Keyframes**: The accuracy depends on selecting keyframes effectively, especially in dynamic scenes.
- **Need for Initialization**: SVO requires an initial bootstrapping phase, which may not be feasible in every scenario.

## Conclusion

SVO is an effective SLAM method that provides a balanced approach to visual odometry. Its hybridization of feature-based and direct methods offers flexibility across various conditions, making it ideal for applications in embedded robotics and drones.

---

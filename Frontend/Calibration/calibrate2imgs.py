import numpy as np
import cv2
import os

def load_calib(calib_file):
    """Load calibration data from KITTI format file"""
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    # Get both P2 and P3 (left and right camera) projection matrices
    P2 = np.array([float(x) for x in lines[2].strip().split()[1:13]]).reshape(3, 4)
    P3 = np.array([float(x) for x in lines[3].strip().split()[1:13]]).reshape(3, 4)
    
    # Extract K, R, and t from P2 (left camera)
    K = P2[:3, :3]
    
    # Get ground truth transformation from P2 to P3
    # P3 = K[R|t] where [R|t] is the transformation from left to right camera
    R_gt = np.linalg.inv(K) @ P3[:, :3]
    t_gt = np.linalg.inv(K) @ P3[:, 3].reshape(3, 1)
    
    return K, R_gt, t_gt

def evaluate_pose(R_pred, t_pred, R_gt, t_gt):
    """Evaluate the predicted pose against ground truth"""
    # Calculate rotation error (in degrees)
    R_error = np.arccos((np.trace(R_pred.T @ R_gt) - 1) / 2) * 180 / np.pi
    
    # Normalize translations for comparison
    t_pred_norm = t_pred / np.linalg.norm(t_pred)
    t_gt_norm = t_gt / np.linalg.norm(t_gt)
    
    # Calculate translation error (in degrees)
    t_error = np.arccos(np.clip(np.dot(t_pred_norm.T, t_gt_norm)[0], -1.0, 1.0)) * 180 / np.pi
    
    # Convert to scalar values
    return float(R_error), float(t_error)

def calculate_relative_pose(img1_path, img2_path, K):
    """Calculate relative pose between two images"""
    # Read images
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Get matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Calculate essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Recover relative rotation and translation
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    
    return R, t

def main():
    # Load camera calibration and ground truth transformation
    calib_file = "../../../KITTI/camera/03/calib.txt"
    K, R_gt, t_gt = load_calib(calib_file)
    
    # Image paths
    img1_path = "../../../KITTI/camera/03/image_0/000000.png"
    img2_path = "../../../KITTI/camera/03/image_1/000000.png"
    
    # Calculate predicted transformation
    R_pred, t_pred = calculate_relative_pose(img1_path, img2_path, K)
    
    # Evaluate predictions
    R_error, t_error = evaluate_pose(R_pred, t_pred, R_gt, t_gt)
    
    print("Predicted Rotation matrix:")
    print(R_pred)
    print("\nGround Truth Rotation matrix:")
    print(R_gt)
    print("\nPredicted Translation vector:")
    print(t_pred)
    print("\nGround Truth Translation vector:")
    print(t_gt)
    print(f"\nRotation Error: {R_error:.2f} degrees")
    print(f"Translation Error: {t_error:.2f} degrees")

if __name__ == "__main__":
    main()

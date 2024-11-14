import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rerun as rr 

def load_calibration(calib_path):
    """Load camera calibration from KITTI calibration file"""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    # Parse projection matrices
    P0 = np.array(list(map(float, lines[0].strip().split()[1:]))).reshape(3,4)
    P1 = np.array(list(map(float, lines[1].strip().split()[1:]))).reshape(3,4)
    
    # Extract intrinsic matrix (same for both cameras)
    K = P0[:, :3]
    
    # First camera is at origin
    R1 = np.eye(3)
    t1 = np.zeros((3, 1))
    
    # Extract R, t for second camera from P1
    R2 = P1[:, :3] @ np.linalg.inv(K)
    t2 = np.linalg.inv(K) @ P1[:, 3].reshape(3, 1)
    
    return K, (R1, t1), (R2, t2)

def find_matches(img1, img2):
    """Find matching points between two images using SIFT"""
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    pts1 = []
    pts2 = []
    
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    
    return np.float32(pts1), np.float32(pts2), good_matches

def triangulate_points(K, R1, t1, R2, t2, pts1, pts2):
    """Triangulate 3D points from corresponding image points"""
    # Compute projection matrices
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))
    
    # Triangulate
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert to 3D points
    pts3D = pts4D[:3, :] / pts4D[3, :]
    return pts3D.T

def visualize_matplotlib(img1, img2, pts1, pts2, matches, points3D, t2):
    """Visualize results using matplotlib"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot first image with keypoints
    ax1 = fig.add_subplot(131)
    ax1.imshow(img1, cmap='gray')
    ax1.scatter(pts1[:, 0], pts1[:, 1], c='r', s=1)
    ax1.set_title('Image 1 Keypoints')
    
    # Plot second image with keypoints
    ax2 = fig.add_subplot(132)
    ax2.imshow(img2, cmap='gray')
    ax2.scatter(pts2[:, 0], pts2[:, 1], c='r', s=1)
    ax2.set_title('Image 2 Keypoints')
    
    # Plot 3D points
    ax3 = fig.add_subplot(133, projection='3d')
    scatter = ax3.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2],
                         c=points3D[:, 2], cmap='viridis', marker='.')
    ax3.set_title('Triangulated 3D Points')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax3, label='Depth')
    
    # Set equal aspect ratio for 3D plot
    ax3.set_box_aspect([1,1,1])
    
    # Add camera positions
    camera_positions = np.array([[0, 0, 0], t2.ravel()])
    ax3.scatter(camera_positions[:, 0], camera_positions[:, 1], 
                camera_positions[:, 2], c='r', marker='^', s=100)
    
    # Add labels
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def visualize_rerun(img1, img2, pts1, pts2, points3D, t2):
    """Visualize results using rerun.io"""
    rr.init("Stereo Triangulation", spawn=True)
    
    # Log left image and its keypoints
    rr.log("world/left_image", rr.Image(img1))
    rr.log("world/left_image/keypoints", rr.Points2D(pts1))
    
    # Log right image and its keypoints
    rr.log("world/right_image", rr.Image(img2))
    rr.log("world/right_image/keypoints", rr.Points2D(pts2))
    
    # Log 3D points in the world space
    rr.log("world/points", 
           rr.Points3D(positions=points3D, colors=np.zeros_like(points3D) + [0.0, 0.5, 1.0]))
    
    # Log camera positions
    camera_positions = np.array([[0, 0, 0], t2.ravel()])
    rr.log("world/cameras", 
           rr.Points3D(positions=camera_positions, colors=np.zeros_like(camera_positions) + [1.0, 0.0, 0.0]))

def main():
    # Load images
    img1 = cv2.imread('../../../KITTI/camera/03/image_0/000000.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../../../KITTI/camera/03/image_1/000000.png', cv2.IMREAD_GRAYSCALE)
    
    # Load calibration
    K, (R1, t1), (R2, t2) = load_calibration('../../../KITTI/camera/03/calib.txt')
    
    # Find matching points
    pts1, pts2, matches = find_matches(img1, img2)
    
    # Triangulate 3D points
    points3D = triangulate_points(K, R1, t1, R2, t2, pts1, pts2)
    
    # Filter out points that are too far or have negative depth
    mask = (np.abs(points3D) < 100).all(axis=1) & (points3D[:, 2] > 0)
    points3D = points3D[mask]
    pts1 = pts1[mask]
    pts2 = pts2[mask]
    
    # Visualize
    visualize_rerun(img1, img2, pts1, pts2, points3D, t2)
    
    print("\nVisualization data has been saved to 'stereo_recording.rrd'")
    print("To view it, open a new terminal and run:")
    print("rerun stereo_recording.rrd")

if __name__ == "__main__":
    main()

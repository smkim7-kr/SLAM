import numpy as np
import cv2
import matplotlib.pyplot as plt

def read_calib_file(filepath):
    """
    Read the calibration file and parse into a dictionary of numpy arrays.
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            value = np.array([float(x) for x in value.strip().split()])
            if key.startswith('P'):
                data[key] = value.reshape(3, 4)
            elif key == 'R0_rect':
                data[key] = value.reshape(3, 3)
            elif key == 'Tr_velo_to_cam':
                data[key] = value.reshape(3, 4)
    return data

def load_velo_scan(velodyne_bin_path):
    """
    Load and parse the velodyne binary file.
    """
    scan = np.fromfile(velodyne_bin_path, dtype=np.float32).reshape(-1, 4)
    return scan

def project_lidar_to_camera_frame(points, calib):
    """
    Project LIDAR points to the camera coordinate system.
    """
    # Extract calibration matrices
    P0 = calib['P0']
    P1 = calib['P1']
    R0_rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']

    # Prepare transformation matrices
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))
    R0_rect = np.hstack((R0_rect, np.zeros((3, 1))))
    R0_rect = np.vstack((R0_rect, [0, 0, 0, 1]))

    # Convert LIDAR points to homogeneous coordinates
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))

    # Transform to camera coordinates
    points_cam = R0_rect @ Tr_velo_to_cam @ points_hom.T

    return points_cam

def project_points_to_image(points_cam, P):
    """
    Project 3D points onto the image plane using projection matrix P.
    """
    points_2d = P @ points_cam[:3, :]
    points_2d = points_2d / points_2d[2, :]
    return points_2d[:2, :].T

def main():
    # File paths
    calib_file = '../../../KITTI/calib/000004.txt'
    velodyne_file = '../../../KITTI/lidar/bin/000004.bin'
    image_0_file = '../../../KITTI/camera/04/image_0/000000.png'
    image_1_file = '../../../KITTI/camera/04/image_1/000000.png'

    # Read calibration data
    calib = read_calib_file(calib_file)

    # Load LIDAR point cloud
    points = load_velo_scan(velodyne_file)

    # Project LIDAR points to camera frame
    points_cam = project_lidar_to_camera_frame(points, calib)

    # Project to image planes
    points_img0 = project_points_to_image(points_cam, calib['P0'])
    points_img1 = project_points_to_image(points_cam, calib['P1'])

    # Load images
    img0 = cv2.imread(image_0_file)
    img1 = cv2.imread(image_1_file)

    # Convert images to RGB
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Filter points within image boundaries
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape

    valid_idx0 = (points_img0[:, 0] >= 0) & (points_img0[:, 0] < w0) & (points_img0[:, 1] >= 0) & (points_img0[:, 1] < h0)
    valid_idx1 = (points_img1[:, 0] >= 0) & (points_img1[:, 0] < w1) & (points_img1[:, 1] >= 0) & (points_img1[:, 1] < h1)

    # Draw projected points on images
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.imshow(img0)
    plt.scatter(points_img0[valid_idx0, 0], points_img0[valid_idx0, 1], c='r', s=1)
    plt.title('Image 0 with projected LIDAR points')

    plt.subplot(1, 2, 2)
    plt.imshow(img1)
    plt.scatter(points_img1[valid_idx1, 0], points_img1[valid_idx1, 1], c='r', s=1)
    plt.title('Image 1 with projected LIDAR points')

    plt.show()

if __name__ == '__main__':
    main()

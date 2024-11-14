import open3d as o3d
import numpy as np

def load_kitti_bin_file(bin_file):
    """
    Load point cloud from a KITTI .bin file and include reflectance.

    Args:
        bin_file (str): Path to the .bin file.

    Returns:
        o3d.geometry.PointCloud: Open3D PointCloud object with colors set based on reflectance.
    """
    # Each point in the .bin file is structured as [x, y, z, reflectance]
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    points = point_cloud[:, :3]  # Extract x, y, z coordinates
    reflectance = point_cloud[:, 3]  # Extract reflectance values

    # Normalize reflectance to range [0, 1] for grayscale coloring
    reflectance_norm = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
    colors = np.stack([reflectance_norm]*3, axis=1)  # Create grayscale colors

    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def detect_ground_plane(pcd, distance_threshold=0.2, ransac_n=3, num_iterations=1000):
    """
    Detect the ground plane using RANSAC.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        distance_threshold (float): RANSAC distance threshold.
        ransac_n (int): Minimum number of points to estimate a plane.
        num_iterations (int): Number of RANSAC iterations.

    Returns:
        plane_model (list): Plane equation coefficients [a, b, c, d] for the plane ax + by + cz + d = 0.
        inliers (list): Indices of the inlier points that belong to the plane.
        outliers (list): Indices of the outlier points that do not belong to the plane.
    """
    # Perform plane segmentation using RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    outliers = list(set(range(len(pcd.points))) - set(inliers))
    return plane_model, inliers, outliers

def visualize_ground_detection(pcd, inliers, outliers, point_size=1.0):
    """
    Visualize the point cloud with ground plane detected.

    Args:
        pcd (o3d.geometry.PointCloud): Original point cloud.
        inliers (list): Indices of the inlier points (ground plane).
        outliers (list): Indices of the outlier points (non-ground).
        point_size (float): Size of the points in the visualization.
    """
    # Extract ground and non-ground points
    ground_pcd = pcd.select_by_index(inliers)
    non_ground_pcd = pcd.select_by_index(outliers)

    # Assign colors
    ground_pcd.paint_uniform_color([0, 1, 0])      # Green for ground
    non_ground_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for non-ground

    # Combine the point clouds
    combined_pcd = ground_pcd + non_ground_pcd

    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Ground Plane Detection', width=800, height=600)

    # Add the combined point cloud
    vis.add_geometry(combined_pcd)

    # Get the render options and set the point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Path to the KITTI .bin file
    bin_file = '../../../../KITTI/lidar/bin/000004.bin'

    # Load the point cloud with grayscale colors based on reflectance
    pcd = load_kitti_bin_file(bin_file)

    # Downsample the point cloud for faster processing (optional)
    voxel_size = 0.1  # Adjust voxel size as needed
    pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Detect the ground plane
    distance_threshold = 0.2  # Maximum distance to the plane for a point to be considered an inlier
    ransac_n = 3              # Number of points to sample for plane fitting
    num_iterations = 1000     # Number of RANSAC iterations
    plane_model, inliers, outliers = detect_ground_plane(pcd_downsampled, distance_threshold, ransac_n, num_iterations)

    # Print the plane equation
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Visualize the ground plane detection
    visualize_ground_detection(pcd_downsampled, inliers, outliers, point_size=2.0)

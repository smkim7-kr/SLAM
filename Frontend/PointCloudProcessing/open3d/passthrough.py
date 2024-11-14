import open3d as o3d
import numpy as np

def load_kitti_bin_file(bin_file):
    """
    Load point cloud from a KITTI .bin file.

    Args:
        bin_file (str): Path to the .bin file.

    Returns:
        numpy.ndarray: Numpy array of shape (N, 3) containing the point cloud.
    """
    # Each point in the .bin file is structured as [x, y, z, reflectance]
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    points = point_cloud[:, :3]  # Extract x, y, z coordinates
    return points

def passthrough_filter(points, min_distance):
    """
    Remove points within a certain radius around the ego-vehicle.

    Args:
        points (numpy.ndarray): Original point cloud data.
        min_distance (float): Minimum distance from the origin to keep points.

    Returns:
        numpy.ndarray: Filtered point cloud data.
    """
    # Calculate distances in the XY plane
    distances = np.linalg.norm(points[:, :2], axis=1)
    # Create a mask for points beyond the minimum distance
    mask = distances > min_distance
    filtered_points = points[mask]
    return filtered_points

def visualize_point_clouds(original_points, filtered_points, point_size=1.0):
    """
    Visualize the original and filtered point clouds for comparison.

    Args:
        original_points (numpy.ndarray): Original point cloud data.
        filtered_points (numpy.ndarray): Filtered point cloud data.
        point_size (float): Size of the points in the visualization.
    """
    # Create Open3D point cloud objects
    pcd_original = o3d.geometry.PointCloud()
    pcd_original.points = o3d.utility.Vector3dVector(original_points)
    pcd_original.paint_uniform_color([1, 0, 0])  # Red color for original

    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    pcd_filtered.paint_uniform_color([0, 0, 1])  # Blue color for filtered

    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Original (Red) vs Filtered (Blue)', width=800, height=600)

    # Add both point clouds to the visualizer
    vis.add_geometry(pcd_original)
    vis.add_geometry(pcd_filtered)

    # Get the render options and set the point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size  # Set your desired point size

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Path to the KITTI .bin file
    bin_file = '../../../../KITTI/lidar/bin/000004.bin'

    # Load the original point cloud
    original_points = load_kitti_bin_file(bin_file)

    # Apply the passthrough filter to remove noise around the ego-vehicle
    min_distance = 2.0  # Adjust the radius as needed
    filtered_points = passthrough_filter(original_points, min_distance)

    # Visualize the original and filtered point clouds with reduced point size
    visualize_point_clouds(original_points, filtered_points, point_size=3.0)

import open3d as o3d
import numpy as np

def load_kitti_bin_file(bin_file):
    """
    Load point cloud from a KITTI .bin file.

    Args:
        bin_file (str): Path to the .bin file.

    Returns:
        o3d.geometry.PointCloud: Open3D PointCloud object.
    """
    # Each point in the .bin file is structured as [x, y, z, reflectance]
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    points = point_cloud[:, :3]  # Extract x, y, z coordinates
    # Create Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def statistical_outlier_removal(pcd, nb_neighbors, std_ratio):
    """
    Apply statistical outlier removal to the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Original point cloud.
        nb_neighbors (int): Number of neighbors to analyze for each point.
        std_ratio (float): Standard deviation multiplier.

    Returns:
        o3d.geometry.PointCloud: Filtered point cloud with outliers removed.
    """
    # Apply statistical outlier removal
    filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                       std_ratio=std_ratio)
    return filtered_pcd, ind

def visualize_point_clouds(pcd_original, pcd_filtered, point_size=1.0):
    """
    Visualize the original and filtered point clouds for comparison.

    Args:
        pcd_original (o3d.geometry.PointCloud): Original point cloud.
        pcd_filtered (o3d.geometry.PointCloud): Filtered point cloud.
        point_size (float): Size of the points in the visualization.
    """
    # Assign colors
    pcd_original.paint_uniform_color([1, 0, 0])  # Red
    pcd_filtered.paint_uniform_color([0, 1, 0])  # Green

    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Statistical Outlier Removal', width=800, height=600)

    # Add both point clouds to the visualizer
    vis.add_geometry(pcd_original)
    vis.add_geometry(pcd_filtered)

    # Get the render options and set the point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size  # Adjust point size as needed

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def visualize_point_cloud(pcd, color, point_size=1.0, window_name='Point Cloud'):
    """
    Visualize a single point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to visualize.
        color (list): RGB color for the point cloud.
        point_size (float): Size of the points in the visualization.
        window_name (str): Name of the visualization window.
    """
    # Assign color to point cloud
    pcd.paint_uniform_color(color)

    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)

    # Add point cloud to the visualizer
    vis.add_geometry(pcd)

    # Get the render options and set the point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size  # Adjust point size as needed

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Path to the KITTI .bin file
    bin_file = '../../../../KITTI/lidar/bin/000004.bin'

    # Load the original point cloud
    original_pcd = load_kitti_bin_file(bin_file)

    # Apply statistical outlier removal
    nb_neighbors = 100    # Number of neighbors to analyze for each point
    std_ratio = 10.0      # Standard deviation multiplier
    filtered_pcd, ind = statistical_outlier_removal(original_pcd, nb_neighbors, std_ratio)

    # Optionally, print the number of points before and after filtering
    print(f"Original point cloud has {len(original_pcd.points)} points.")
    print(f"Filtered point cloud has {len(filtered_pcd.points)} points.")

    # Visualize the original and filtered point clouds together
    visualize_point_clouds(original_pcd, filtered_pcd, point_size=3.0)

    # # Alternatively, visualize separately
    # # Visualize original point cloud
    # visualize_point_cloud(original_pcd, color=[1, 0, 0], point_size=1.0, window_name='Original Point Cloud')

    # # Visualize filtered point cloud
    # visualize_point_cloud(filtered_pcd, color=[0, 1, 0], point_size=1.0, window_name='Filtered Point Cloud')

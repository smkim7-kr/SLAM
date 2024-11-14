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

def downsample_point_cloud(pcd, voxel_size):
    """
    Downsample the point cloud using voxel downsampling.

    Args:
        pcd (o3d.geometry.PointCloud): Original point cloud.
        voxel_size (float): Voxel size for downsampling.

    Returns:
        o3d.geometry.PointCloud: Downsampled point cloud.
    """
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd

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

    # Downsample the point cloud
    voxel_size_1 = 0.2  # Adjust the voxel size as needed
    downsampled_pcd = downsample_point_cloud(original_pcd, voxel_size=voxel_size_1)

    # Double downsample the point cloud (downsample the downsampled point cloud)
    voxel_size_2 = 0.4  # Adjust the voxel size as needed
    double_downsampled_pcd = downsample_point_cloud(downsampled_pcd, voxel_size=voxel_size_2)

    # Optionally, print the number of points in each point cloud
    print(f"Original point cloud has {len(original_pcd.points)} points.")
    print(f"First downsampled point cloud has {len(downsampled_pcd.points)} points.")
    print(f"Second downsampled point cloud has {len(double_downsampled_pcd.points)} points.")

    # Visualize each point cloud separately
    # Original point cloud
    visualize_point_cloud(original_pcd, color=[1, 0, 0], point_size=3.0, window_name='Original Point Cloud')

    # First downsampled point cloud
    visualize_point_cloud(downsampled_pcd, color=[0, 1, 0], point_size=3.0, window_name='First Downsampled Point Cloud')

    # Second downsampled point cloud
    visualize_point_cloud(double_downsampled_pcd, color=[0, 0, 1], point_size=3.0, window_name='Second Downsampled Point Cloud')

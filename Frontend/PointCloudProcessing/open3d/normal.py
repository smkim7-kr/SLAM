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

def estimate_normals(pcd, radius, max_nn):
    """
    Estimate normals for the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud.
        radius (float): Search radius for neighborhood.
        max_nn (int): Maximum number of neighbors to consider.

    Returns:
        o3d.geometry.PointCloud: Point cloud with estimated normals.
    """
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    # Optionally, normalize the normals
    pcd.normalize_normals()
    return pcd

def visualize_point_cloud_with_normals(pcd, point_size=1.0, normal_length=0.5):
    """
    Visualize the point cloud with normals.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud with normals estimated.
        point_size (float): Size of the points in the visualization.
        normal_length (float): Length of the normal vectors to be displayed.
    """
    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud with Normals', width=800, height=600)

    # Add the point cloud
    vis.add_geometry(pcd)

    # Get the render options and set the point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    # Create a LineSet for normals
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    lines = []

    # Sample a subset of points to display normals (to avoid clutter)
    sample_indices = np.random.choice(len(points), size=500, replace=False)
    line_points = []
    line_indices = []
    for i, idx in enumerate(sample_indices):
        point = points[idx]
        normal = normals[idx]
        line_points.append(point)
        line_points.append(point + normal * normal_length)
        line_indices.append([2 * i, 2 * i + 1])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(line_indices))])  # Red normals

    # Add the line set to the visualizer
    vis.add_geometry(line_set)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Path to the KITTI .bin file
    bin_file = '../../../../KITTI/lidar/bin/000004.bin'

    # Load the point cloud with grayscale colors based on reflectance
    pcd = load_kitti_bin_file(bin_file)

    # Estimate normals
    radius = 1.0  # Adjust the search radius as needed
    max_nn = 30   # Maximum number of neighbors to consider
    pcd = estimate_normals(pcd, radius, max_nn)

    # Optionally, orient the normals towards a viewpoint
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    # Visualize the point cloud with normals
    visualize_point_cloud_with_normals(pcd, point_size=1.0, normal_length=0.5)

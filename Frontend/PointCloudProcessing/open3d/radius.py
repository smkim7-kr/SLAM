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

def build_kdtree(pcd):
    """
    Build a KDTree from the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud.

    Returns:
        o3d.geometry.KDTreeFlann: KDTree built from the point cloud.
    """
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    return kdtree

def radius_neighbor_search(kdtree, query_point, radius):
    """
    Perform radius neighbor search.

    Args:
        kdtree (o3d.geometry.KDTreeFlann): KDTree.
        query_point (numpy.ndarray): Query point (1x3).
        radius (float): Radius within which to search for neighbors.

    Returns:
        indices (list): Indices of the neighbors within the radius.
    """
    # Perform search
    [_, idx, _] = kdtree.search_radius_vector_3d(query_point, radius)
    return idx

def visualize_neighbors(pcd, query_point, neighbor_indices, point_size=1.0):
    """
    Visualize the point cloud and highlight the query point and its neighbors within the radius.

    Args:
        pcd (o3d.geometry.PointCloud): Original point cloud.
        query_point (numpy.ndarray): Query point (1x3).
        neighbor_indices (list): Indices of the neighbors within the radius.
        point_size (float): Size of the points in the visualization.
    """
    # Create copies to avoid modifying the original point cloud
    pcd_copy = pcd.select_by_index(neighbor_indices, invert=True)
    neighbors_pcd = pcd.select_by_index(neighbor_indices)

    # Create a PointCloud for the query point
    query_pcd = o3d.geometry.PointCloud()
    query_pcd.points = o3d.utility.Vector3dVector([query_point])

    # Assign colors
    pcd_copy.paint_uniform_color([0.8, 0.8, 0.8])    # Gray for the rest of the cloud
    neighbors_pcd.paint_uniform_color([0, 0, 1])     # Blue for neighbors
    query_pcd.paint_uniform_color([1, 0, 0])         # Red for the query point

    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Radius Neighbor Search', width=800, height=600)

    # Add geometries to the visualizer
    vis.add_geometry(pcd_copy)
    vis.add_geometry(neighbors_pcd)
    vis.add_geometry(query_pcd)

    # Get the render options and set the point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size

    # Run the visualizer
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Path to the KITTI .bin file
    bin_file = '../../../../KITTI/lidar/bin/000004.bin'

    # Load the point cloud
    pcd = load_kitti_bin_file(bin_file)

    # Build KDTree
    kdtree = build_kdtree(pcd)

    # Define a query point (e.g., choose a random point from the point cloud)
    points = np.asarray(pcd.points)
    query_point = points[np.random.choice(len(points))]

    # Perform radius neighbor search
    radius = 1.0  # Radius within which to search for neighbors (adjust as needed)
    neighbor_indices = radius_neighbor_search(kdtree, query_point, radius)

    # Visualize the query point and its neighbors within the radius
    visualize_neighbors(pcd, query_point, neighbor_indices, point_size=3.0)

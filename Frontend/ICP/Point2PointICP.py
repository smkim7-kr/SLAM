import numpy as np
import open3d as o3d
import copy

# Load the point cloud from the .bin file
points = np.fromfile('../../../KITTI/lidar/bin/000004.bin', dtype=np.float32).reshape(-1, 4)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

# First downsampling
voxel_size1 = 0.3  # Increased from 0.1
pcd_down1 = pcd.voxel_down_sample(voxel_size=voxel_size1)

# Second downsampling
voxel_size2 = 0.5  # Increased from 0.2
pcd_down2 = pcd_down1.voxel_down_sample(voxel_size=voxel_size2)

# Apply a transformation to create the destination point cloud
transformation = np.array([
    [0.862, 0.011, -0.507, 0.5],
    [-0.139, 0.967, -0.215, 0.7],
    [0.487, 0.255, 0.835, -1.4],
    [0.0,   0.0,   0.0,    1.0]
])

# Copy and transform the original point cloud
pcd_transformed = copy.deepcopy(pcd_down2)
pcd_transformed.transform(transformation)

# Perform point-to-point ICP
trans_init = np.identity(4)  # Initial transformation guess (identity matrix)
threshold = 0.5  # Increased threshold for correspondence pairs
max_iteration = 100  # Add maximum iteration parameter
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd_down2, pcd_transformed,
    threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

print(reg_p2p)
print(f"Ground truth transformation:\n{transformation}")
print(f"ICP transformation:\n{reg_p2p.transformation}")

# Apply the ICP result to align the source to target
pcd_aligned = copy.deepcopy(pcd_down2)
pcd_aligned.transform(reg_p2p.transformation)

# Paint the point clouds with different colors
pcd_down2.paint_uniform_color([1, 0, 0])          # Source: Red
pcd_transformed.paint_uniform_color([0, 1, 0])     # Target: Green
pcd_aligned.paint_uniform_color([0, 0, 1])         # Aligned Result: Blue

# Visualize all point clouds
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the geometries
vis.add_geometry(pcd_down2)
vis.add_geometry(pcd_transformed)
vis.add_geometry(pcd_aligned)

# Improve visualization settings
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])  # Black background
opt.point_size = 2.0  # Larger points

vis.run()
vis.destroy_window()

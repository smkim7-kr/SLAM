import numpy as np
import open3d as o3d
import copy

# Load and prepare point clouds
points = np.fromfile('../../../KITTI/lidar/bin/000004.bin', dtype=np.float32).reshape(-1, 4)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

# First downsampling
voxel_size1 = 0.3
pcd_down1 = pcd.voxel_down_sample(voxel_size=voxel_size1)

# Second downsampling
voxel_size2 = 0.5
pcd_down2 = pcd_down1.voxel_down_sample(voxel_size=voxel_size2)

# Apply a transformation to create the destination point cloud
transformation = np.array([
    [0.862, 0.011, -0.507, 0.5],
    [-0.139, 0.967, -0.215, 0.7],
    [0.487, 0.255, 0.835, -1.4],
    [0.0,   0.0,   0.0,    1.0]
])

# Challenging transformation (larger rotation and translation)
# GICP fails to converge with this transformation
# transformation = np.array([
#     [ 0.342,  0.883, -0.321,  2.5],  # Larger translation in x
#     [-0.892,  0.235,  0.387,  1.8],  # Larger translation in y
#     [ 0.293,  0.407,  0.865, -3.2],  # Larger translation in z
#     [ 0.0,    0.0,    0.0,    1.0]
# ])

pcd_transformed = copy.deepcopy(pcd_down2)
pcd_transformed.transform(transformation)

# Estimate normals for both point clouds
radius = voxel_size2 * 2
pcd_down2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
pcd_transformed.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

# Perform Generalized ICP
trans_init = np.identity(4)
threshold = 0.5
max_iteration = 100

reg_gicp = o3d.pipelines.registration.registration_generalized_icp(
    pcd_down2, pcd_transformed,
    threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

print(f"Ground truth transformation:\n{transformation}")
print(f"GICP transformation:\n{reg_gicp.transformation}")
print(f"Fitness: {reg_gicp.fitness}")
print(f"Inlier RMSE: {reg_gicp.inlier_rmse}")

# Apply the GICP result
pcd_aligned = copy.deepcopy(pcd_down2)
pcd_aligned.transform(reg_gicp.transformation)

# Visualization
pcd_down2.paint_uniform_color([1, 0, 0])          # Source: Red
pcd_transformed.paint_uniform_color([0, 1, 0])     # Target: Green
pcd_aligned.paint_uniform_color([0, 0, 1])         # Aligned Result: Blue

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(pcd_down2)
vis.add_geometry(pcd_transformed)
vis.add_geometry(pcd_aligned)

opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.point_size = 2.0

vis.run()
vis.destroy_window()

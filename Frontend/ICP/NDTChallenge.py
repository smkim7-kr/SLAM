import numpy as np
import open3d as o3d
import copy

# Load and prepare point clouds
points = np.fromfile('../../../KITTI/lidar/bin/000004.bin', dtype=np.float32).reshape(-1, 4)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])

# Multi-scale downsampling
voxel_size_coarse = 1.0  # Coarse level
voxel_size_medium = 0.5  # Medium level
voxel_size_fine = 0.3    # Fine level

pcd_down_coarse = pcd.voxel_down_sample(voxel_size=voxel_size_coarse)
pcd_down_medium = pcd.voxel_down_sample(voxel_size=voxel_size_medium)
pcd_down_fine = pcd.voxel_down_sample(voxel_size=voxel_size_fine)

# Challenging transformation
transformation = np.array([
    [ 0.342,  0.883, -0.321,  2.5],  # Larger translation in x
    [-0.892,  0.235,  0.387,  1.8],  # Larger translation in y
    [ 0.293,  0.407,  0.865, -3.2],  # Larger translation in z
    [ 0.0,    0.0,    0.0,    1.0]
])

# Create transformed versions for each scale
pcd_transformed_coarse = copy.deepcopy(pcd_down_coarse)
pcd_transformed_medium = copy.deepcopy(pcd_down_medium)
pcd_transformed_fine = copy.deepcopy(pcd_down_fine)

pcd_transformed_coarse.transform(transformation)
pcd_transformed_medium.transform(transformation)
pcd_transformed_fine.transform(transformation)

# Compute FPFH features for coarse global registration
def prepare_dataset(source, target, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5
    
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return source_fpfh, target_fpfh

# Global registration using FPFH features
source_fpfh, target_fpfh = prepare_dataset(pcd_down_coarse, pcd_transformed_coarse, voxel_size_coarse)

result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    pcd_down_coarse, pcd_transformed_coarse, source_fpfh, target_fpfh,
    mutual_filter=True,
    max_correspondence_distance=voxel_size_coarse * 1.5,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    ransac_n=4,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size_coarse * 1.5)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

# Multi-scale refinement using ICP
def refine_registration(source, target, voxel_size, init_transform):
    # Estimate normals for both source and target
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=200
    )
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=voxel_size * 1.5,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=criteria
    )
    return reg_p2p

# Multi-scale refinement
current_transform = result_ransac.transformation
reg_coarse = refine_registration(pcd_down_coarse, pcd_transformed_coarse, 
                                voxel_size_coarse, current_transform)

current_transform = reg_coarse.transformation
reg_medium = refine_registration(pcd_down_medium, pcd_transformed_medium, 
                                voxel_size_medium, current_transform)

current_transform = reg_medium.transformation
reg_fine = refine_registration(pcd_down_fine, pcd_transformed_fine, 
                              voxel_size_fine, current_transform)

# Final result
print(f"Ground truth transformation:\n{transformation}")
print(f"Estimated transformation:\n{reg_fine.transformation}")
print(f"Fitness: {reg_fine.fitness}")
print(f"Inlier RMSE: {reg_fine.inlier_rmse}")

# Visualization
pcd_down_fine.paint_uniform_color([1, 0, 0])          # Source: Red
pcd_transformed_fine.paint_uniform_color([0, 1, 0])   # Target: Green

# Apply the final transformation
pcd_aligned = copy.deepcopy(pcd_down_fine)
pcd_aligned.transform(reg_fine.transformation)
pcd_aligned.paint_uniform_color([0, 0, 1])           # Aligned Result: Blue

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(pcd_down_fine)
vis.add_geometry(pcd_transformed_fine)
vis.add_geometry(pcd_aligned)

opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
opt.point_size = 2.0

vis.run()
vis.destroy_window()
import numpy as np
import open3d as o3d
import copy

# Generate random points on a sphere for source point cloud
def generate_sphere_points(num_points, radius, center):
    phi = np.random.uniform(0, 2*np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    theta = np.arccos(costheta)
    
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1] 
    z = radius * np.cos(theta) + center[2]
    
    return np.column_stack((x, y, z))

# Create source point cloud
num_points = 5000
src_points = generate_sphere_points(num_points, radius=10, center=[0, 0, 0])
pcd_src = o3d.geometry.PointCloud()
pcd_src.points = o3d.utility.Vector3dVector(src_points)

# Create destination point cloud with a transformation
# Generate points for destination (slightly moved sphere)
dst_points = generate_sphere_points(num_points, radius=10, center=[2, 3, -1])
pcd_dst = o3d.geometry.PointCloud()
pcd_dst.points = o3d.utility.Vector3dVector(dst_points)

# First downsampling for both point clouds
voxel_size1 = 0.3
pcd_src_down1 = pcd_src.voxel_down_sample(voxel_size=voxel_size1)
pcd_dst_down1 = pcd_dst.voxel_down_sample(voxel_size=voxel_size1)

# Second downsampling for both point clouds
voxel_size2 = 0.5
pcd_src_down2 = pcd_src_down1.voxel_down_sample(voxel_size=voxel_size2)
pcd_dst_down2 = pcd_dst_down1.voxel_down_sample(voxel_size=voxel_size2)

# Estimate normals for both point clouds
radius = voxel_size2 * 2
pcd_src_down2.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
pcd_dst_down2.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

# Perform point-to-plane ICP
trans_init = np.identity(4)  # Initial transformation guess
threshold = 0.5
max_iteration = 100
reg_p2plane = o3d.pipelines.registration.registration_icp(
    pcd_src_down2, pcd_dst_down2,
    threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))

print(reg_p2plane)
print(f"ICP transformation:\n{reg_p2plane.transformation}")

# Apply the ICP result to align the source to target
pcd_aligned = copy.deepcopy(pcd_src_down2)
pcd_aligned.transform(reg_p2plane.transformation)

# Paint the point clouds with different colors
pcd_src_down2.paint_uniform_color([1, 0, 0])      # Source: Red
pcd_dst_down2.paint_uniform_color([0, 1, 0])      # Target: Green
pcd_aligned.paint_uniform_color([0, 0, 1])        # Aligned Result: Blue

# Visualize all point clouds
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the geometries
vis.add_geometry(pcd_src_down2)
vis.add_geometry(pcd_dst_down2)
vis.add_geometry(pcd_aligned)

# Improve visualization settings
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])  # Black background
opt.point_size = 2.0  # Larger points

vis.run()
vis.destroy_window()

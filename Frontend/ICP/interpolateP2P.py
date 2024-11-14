import numpy as np
import open3d as o3d
import copy
import time

def interpolate_transformation(T1, T2, t):
    """
    Simple linear interpolation between two 4x4 transformation matrices
    t: interpolation parameter [0, 1]
    """
    return T1 + t * (T2 - T1)

def animate_icp(source, target, n_frames=50):
    """
    Animate ICP process from source to target
    n_frames: number of interpolation frames
    """
    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add source point cloud (will be updated)
    pcd_animated = copy.deepcopy(source)
    pcd_animated.paint_uniform_color([1, 0, 0])  # Red
    vis.add_geometry(pcd_animated)
    
    # Add target point cloud (static)
    target.paint_uniform_color([0, 1, 0])  # Green
    vis.add_geometry(target)
    
    # Improve visualization settings
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # Set up camera view
    vc = vis.get_view_control()
    vc.set_zoom(0.8)
    vc.set_front([0, 0, -1])
    vc.set_up([0, -1, 0])
    
    # Perform ICP
    threshold = 0.5
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target,
        threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    print("ICP fitness:", reg_p2p.fitness)
    print("ICP RMSE:", reg_p2p.inlier_rmse)
    
    # Get initial and final transformations
    T_start = np.identity(4)
    T_end = reg_p2p.transformation
    
    # Animate the transformation
    for frame in range(n_frames + 1):
        t = frame / n_frames  # interpolation parameter [0, 1]
        
        # Interpolate transformation
        T_current = interpolate_transformation(T_start, T_end, t)
        
        # Update point cloud position
        pcd_animated.points = copy.deepcopy(source.points)
        pcd_animated.transform(T_current)
        
        # Update visualization
        vis.update_geometry(pcd_animated)
        vis.poll_events()
        vis.update_renderer()
        
        # Add small delay for smooth animation
        time.sleep(0.05)
    
    # Keep window open until key press instead of auto-closing
    print("Press any key to exit...")
    vis.run()  # This will block until window is closed
    vis.destroy_window()

def main():
    # Load and prepare point clouds
    points = np.fromfile('../../../KITTI/lidar/bin/000004.bin', 
                        dtype=np.float32).reshape(-1, 4)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Downsample
    voxel_size1 = 0.3
    pcd_down1 = pcd.voxel_down_sample(voxel_size=voxel_size1)
    voxel_size2 = 0.5
    pcd_down2 = pcd_down1.voxel_down_sample(voxel_size=voxel_size2)
    
    # Create transformed target
    transformation = np.array([
        [0.862, 0.011, -0.507, 0.5],
        [-0.139, 0.967, -0.215, 0.7],
        [0.487, 0.255, 0.835, -1.4],
        [0.0,   0.0,   0.0,    1.0]
    ])
    pcd_transformed = copy.deepcopy(pcd_down2)
    pcd_transformed.transform(transformation)
    
    # Run animation
    animate_icp(pcd_down2, pcd_transformed, n_frames=50)

if __name__ == "__main__":
    main()
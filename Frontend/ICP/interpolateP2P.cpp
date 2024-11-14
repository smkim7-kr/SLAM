#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <thread>

void interpolate_transformation(const Eigen::Matrix4f& T1, const Eigen::Matrix4f& T2, float t, Eigen::Matrix4f& T_out) {
    T_out = T1 + t * (T2 - T1);
}

void animate_icp(pcl::PointCloud<pcl::PointXYZ>::Ptr source, pcl::PointCloud<pcl::PointXYZ>::Ptr target, int n_frames = 50) {
    pcl::visualization::PCLVisualizer viewer("ICP Animation");
    viewer.setBackgroundColor(0, 0, 0);
    
    // Add target point cloud (static)
    viewer.addPointCloud<pcl::PointXYZ>(target, "target cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "target cloud"); // Green

    // Perform ICP
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source);
    icp.setInputTarget(target);
    
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);
    
    std::cout << "ICP fitness: " << icp.getFitnessScore() << std::endl;
    std::cout << "ICP RMSE: " << icp.getFitnessScore() << std::endl;

    // Get initial and final transformations
    Eigen::Matrix4f T_start = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f T_end = icp.getFinalTransformation();

    // Animate the transformation
    for (int frame = 0; frame <= n_frames; ++frame) {
        float t = static_cast<float>(frame) / n_frames; // interpolation parameter [0, 1]
        
        // Interpolate transformation
        Eigen::Matrix4f T_current;
        interpolate_transformation(T_start, T_end, t, T_current);
        
        // Update point cloud position
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_animated(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*source, *pcd_animated, T_current);
        
        if (pcd_animated->points.empty()) {
            std::cerr << "Animated point cloud is empty!" << std::endl;
            continue; // Skip this iteration
        }
        
        // Update visualization
        viewer.removePointCloud("animated cloud");
        viewer.addPointCloud<pcl::PointXYZ>(pcd_animated, "animated cloud");
        
        viewer.spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Add small delay for smooth animation
    }

    // Keep window open until key press
    std::cout << "Press any key to exit..." << std::endl;
    while (!viewer.wasStopped()) {
        viewer.spinOnce();
    }
}

int main() {
    // Load and prepare point clouds
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../../../../KITTI/lidar/pcd/000004.pcd", *pcd) == -1) {
        PCL_ERROR("Couldn't read file \n");
        return -1;
    }

    if (pcd->empty()) {
        std::cerr << "Loaded point cloud is empty!" << std::endl;
        return -1;
    }

    // Downsample
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_down1(new pcl::PointCloud<pcl::PointXYZ>);
    sor.setInputCloud(pcd);
    sor.setLeafSize(0.3f, 0.3f, 0.3f);
    sor.filter(*pcd_down1);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_down2(new pcl::PointCloud<pcl::PointXYZ>);
    sor.setInputCloud(pcd_down1);
    sor.setLeafSize(0.5f, 0.5f, 0.5f);
    sor.filter(*pcd_down2);

    // Create transformed target
    Eigen::Matrix4f transformation;
    transformation << 0.862, 0.011, -0.507, 0.5,
                       -0.139, 0.967, -0.215, 0.7,
                       0.487, 0.255, 0.835, -1.4,
                       0.0,   0.0,   0.0,    1.0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*pcd_down2, *pcd_transformed, transformation);

    // Run animation
    animate_icp(pcd_down2, pcd_transformed, 50);

    return 0;
}
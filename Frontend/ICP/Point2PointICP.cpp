#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv) {
    // Load the point cloud from a .bin file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Load your point cloud data here (you may need to convert .bin to .pcd)
    pcl::io::loadPCDFile("../../../../KITTI/lidar/pcd/000004.pcd", *cloud);

    // First downsampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg1;
    vg1.setInputCloud(cloud);
    vg1.setLeafSize(0.3f, 0.3f, 0.3f);  // Increased from 0.1
    vg1.filter(*cloud_down1);

    // Second downsampling
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_down2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> vg2;
    vg2.setInputCloud(cloud_down1);
    vg2.setLeafSize(0.5f, 0.5f, 0.5f);  // Increased from 0.2
    vg2.filter(*cloud_down2);

    // Apply a transformation to create the destination point cloud
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    transformation(0, 0) = 0.862; transformation(0, 1) = 0.011; transformation(0, 2) = -0.507; transformation(0, 3) = 0.5;
    transformation(1, 0) = -0.139; transformation(1, 1) = 0.967; transformation(1, 2) = -0.215; transformation(1, 3) = 0.7;
    transformation(2, 0) = 0.487; transformation(2, 1) = 0.255; transformation(2, 2) = 0.835; transformation(2, 3) = -1.4;

    // Transform the original point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_down2, *cloud_transformed, transformation);

    // Perform point-to-point ICP
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setMaximumIterations(100);  // Add maximum iteration parameter
    icp.setInputSource(cloud_down2);
    icp.setInputTarget(cloud_transformed);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    // Output the results
    std::cout << "ICP has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
    std::cout << "ICP transformation:\n" << icp.getFinalTransformation() << std::endl;

    // Check if point clouds are valid before visualization
    if (cloud_down2->empty() || cloud_transformed->empty() || Final.empty()) {
        std::cerr << "One of the point clouds is empty!" << std::endl;
        return -1;  // Exit if any point cloud is empty
    }

    // Visualize the point clouds
    pcl::visualization::PCLVisualizer viewer("3D Viewer");
    viewer.setBackgroundColor(0, 0, 0);  // Black background
    viewer.addPointCloud<pcl::PointXYZ>(cloud_down2, "source cloud");
    viewer.addPointCloud<pcl::PointXYZ>(cloud_transformed, "target cloud");
    viewer.addPointCloud<pcl::PointXYZ>(Final.makeShared(), "aligned cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "source cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "target cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "aligned cloud");

    // Spin the viewer until it is closed
    while (!viewer.wasStopped()) {
        viewer.spinOnce(100);
    }

    return 0;
}

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <fstream>
#include <vector>

pcl::PointCloud<pcl::PointXYZ>::Ptr loadKittiBinFile(const std::string& binFile) {
    // Create point cloud object
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Read binary file
    std::ifstream file(binFile, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << binFile << std::endl;
        return cloud;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    const size_t num_elements = file.tellg() / (4 * sizeof(float));
    file.seekg(0, std::ios::beg);

    // Read point cloud data
    std::vector<float> buffer(num_elements * 4);
    file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
    
    // Convert to PCL point cloud
    cloud->points.reserve(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        pcl::PointXYZ point;
        point.x = buffer[i * 4];
        point.y = buffer[i * 4 + 1];
        point.z = buffer[i * 4 + 2];
        // Note: we ignore reflectance (buffer[i * 4 + 3])
        cloud->points.push_back(point);
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr downsamplePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    float voxelSize) {
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(voxelSize, voxelSize, voxelSize);
    voxel_filter.filter(*downsampled_cloud);
    
    return downsampled_cloud;
}

void visualizePointCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<double>& color,
    double pointSize,
    const std::string& windowName) {
    
    // Check if cloud is valid and not empty
    if (!cloud || cloud->empty()) {
        std::cerr << "Warning: Empty or invalid point cloud for " << windowName << ". Skipping visualization." << std::endl;
        return;
    }
    
    // Create visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(windowName));
    viewer->setBackgroundColor(0, 0, 0);
    
    // Add point cloud to viewer
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(
        cloud, color[0] * 255, color[1] * 255, color[2] * 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, pointSize, "cloud");
    
    // Wait until viewer is closed
    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
}

int main(int argc, char** argv) {
    // Path to the KITTI .bin file
    std::string binFile = "../../../../../KITTI/lidar/bin/000004.bin";
    
    // Load the original point cloud
    auto originalCloud = loadKittiBinFile(binFile);
    
    // Check if cloud loading was successful
    if (originalCloud->empty()) {
        std::cerr << "Failed to load point cloud. Exiting." << std::endl;
        return -1;
    }
    
    // Downsample the point cloud
    float voxelSize1 = 0.2f;
    auto downsampledCloud = downsamplePointCloud(originalCloud, voxelSize1);
    
    // Double downsample the point cloud
    float voxelSize2 = 0.4f;
    auto doubleDownsampledCloud = downsamplePointCloud(downsampledCloud, voxelSize2);
    
    // Print point cloud sizes
    std::cout << "Original point cloud has " << originalCloud->size() << " points." << std::endl;
    std::cout << "First downsampled point cloud has " << downsampledCloud->size() << " points." << std::endl;
    std::cout << "Second downsampled point cloud has " << doubleDownsampledCloud->size() << " points." << std::endl;
    
    // Visualize point clouds
    // Original point cloud (red)
    visualizePointCloud(originalCloud, {1.0, 0.0, 0.0}, 3.0, "Original Point Cloud");
    
    // First downsampled point cloud (green)
    visualizePointCloud(downsampledCloud, {0.0, 1.0, 0.0}, 3.0, "First Downsampled Point Cloud");
    
    // Second downsampled point cloud (blue)
    visualizePointCloud(doubleDownsampledCloud, {0.0, 0.0, 1.0}, 3.0, "Second Downsampled Point Cloud");
    
    return 0;
}
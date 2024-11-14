cmake_minimum_required(VERSION 3.10)
project(ICPProject)

# Find PCL package
find_package(PCL 1.8 REQUIRED)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})

# Create the executable
add_executable(ICPInterpolate interpolateP2P.cpp)
add_executable(ICPP2P Point2PointICP.cpp)

# Link the PCL libraries
target_link_libraries(ICPInterpolate ${PCL_LIBRARIES})
target_link_libraries(ICPP2P ${PCL_LIBRARIES})

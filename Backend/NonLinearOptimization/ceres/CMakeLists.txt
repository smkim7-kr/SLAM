cmake_minimum_required(VERSION 3.10)
project(CeresTutorial)

# Find Ceres
include_directories("/usr/include/eigen3")
find_package(Ceres REQUIRED)

# Include directories
include_directories(${CERES_INCLUDE_DIRS})

# Add your executable
add_executable(CeresExample main.cpp)

# Link Ceres to your executable
target_link_libraries(CeresExample ${CERES_LIBRARIES})
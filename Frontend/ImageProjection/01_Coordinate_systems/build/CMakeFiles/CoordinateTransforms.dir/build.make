# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/build

# Include any dependencies generated for this target.
include CMakeFiles/CoordinateTransforms.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CoordinateTransforms.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CoordinateTransforms.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CoordinateTransforms.dir/flags.make

CMakeFiles/CoordinateTransforms.dir/main.cpp.o: CMakeFiles/CoordinateTransforms.dir/flags.make
CMakeFiles/CoordinateTransforms.dir/main.cpp.o: ../main.cpp
CMakeFiles/CoordinateTransforms.dir/main.cpp.o: CMakeFiles/CoordinateTransforms.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CoordinateTransforms.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CoordinateTransforms.dir/main.cpp.o -MF CMakeFiles/CoordinateTransforms.dir/main.cpp.o.d -o CMakeFiles/CoordinateTransforms.dir/main.cpp.o -c /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/main.cpp

CMakeFiles/CoordinateTransforms.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CoordinateTransforms.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/main.cpp > CMakeFiles/CoordinateTransforms.dir/main.cpp.i

CMakeFiles/CoordinateTransforms.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CoordinateTransforms.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/main.cpp -o CMakeFiles/CoordinateTransforms.dir/main.cpp.s

# Object files for target CoordinateTransforms
CoordinateTransforms_OBJECTS = \
"CMakeFiles/CoordinateTransforms.dir/main.cpp.o"

# External object files for target CoordinateTransforms
CoordinateTransforms_EXTERNAL_OBJECTS =

CoordinateTransforms: CMakeFiles/CoordinateTransforms.dir/main.cpp.o
CoordinateTransforms: CMakeFiles/CoordinateTransforms.dir/build.make
CoordinateTransforms: CMakeFiles/CoordinateTransforms.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CoordinateTransforms"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CoordinateTransforms.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CoordinateTransforms.dir/build: CoordinateTransforms
.PHONY : CMakeFiles/CoordinateTransforms.dir/build

CMakeFiles/CoordinateTransforms.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CoordinateTransforms.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CoordinateTransforms.dir/clean

CMakeFiles/CoordinateTransforms.dir/depend:
	cd /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/build /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/build /home/smkim/spatialai_slam_tutorial/_tutorial/04_Camera_images/01_Coordinate_systems/build/CMakeFiles/CoordinateTransforms.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CoordinateTransforms.dir/depend

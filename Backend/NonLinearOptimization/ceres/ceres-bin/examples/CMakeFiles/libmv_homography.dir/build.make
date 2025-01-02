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
CMAKE_SOURCE_DIR = /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin

# Include any dependencies generated for this target.
include examples/CMakeFiles/libmv_homography.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/libmv_homography.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/libmv_homography.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/libmv_homography.dir/flags.make

examples/CMakeFiles/libmv_homography.dir/libmv_homography.cc.o: examples/CMakeFiles/libmv_homography.dir/flags.make
examples/CMakeFiles/libmv_homography.dir/libmv_homography.cc.o: /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/examples/libmv_homography.cc
examples/CMakeFiles/libmv_homography.dir/libmv_homography.cc.o: examples/CMakeFiles/libmv_homography.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/libmv_homography.dir/libmv_homography.cc.o"
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/libmv_homography.dir/libmv_homography.cc.o -MF CMakeFiles/libmv_homography.dir/libmv_homography.cc.o.d -o CMakeFiles/libmv_homography.dir/libmv_homography.cc.o -c /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/examples/libmv_homography.cc

examples/CMakeFiles/libmv_homography.dir/libmv_homography.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libmv_homography.dir/libmv_homography.cc.i"
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/examples/libmv_homography.cc > CMakeFiles/libmv_homography.dir/libmv_homography.cc.i

examples/CMakeFiles/libmv_homography.dir/libmv_homography.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libmv_homography.dir/libmv_homography.cc.s"
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/examples/libmv_homography.cc -o CMakeFiles/libmv_homography.dir/libmv_homography.cc.s

# Object files for target libmv_homography
libmv_homography_OBJECTS = \
"CMakeFiles/libmv_homography.dir/libmv_homography.cc.o"

# External object files for target libmv_homography
libmv_homography_EXTERNAL_OBJECTS =

bin/libmv_homography: examples/CMakeFiles/libmv_homography.dir/libmv_homography.cc.o
bin/libmv_homography: examples/CMakeFiles/libmv_homography.dir/build.make
bin/libmv_homography: lib/libceres.a
bin/libmv_homography: /usr/lib/x86_64-linux-gnu/libglog.so.0.4.0
bin/libmv_homography: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
bin/libmv_homography: /usr/lib/x86_64-linux-gnu/libunwind.so
bin/libmv_homography: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/libmv_homography: /usr/lib/x86_64-linux-gnu/libblas.so
bin/libmv_homography: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/libmv_homography: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/libmv_homography: examples/CMakeFiles/libmv_homography.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/libmv_homography"
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libmv_homography.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/libmv_homography.dir/build: bin/libmv_homography
.PHONY : examples/CMakeFiles/libmv_homography.dir/build

examples/CMakeFiles/libmv_homography.dir/clean:
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/examples && $(CMAKE_COMMAND) -P CMakeFiles/libmv_homography.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/libmv_homography.dir/clean

examples/CMakeFiles/libmv_homography.dir/depend:
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0 /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/examples /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/examples /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/examples/CMakeFiles/libmv_homography.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/libmv_homography.dir/depend

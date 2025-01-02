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
include internal/ceres/CMakeFiles/is_close_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include internal/ceres/CMakeFiles/is_close_test.dir/compiler_depend.make

# Include the progress variables for this target.
include internal/ceres/CMakeFiles/is_close_test.dir/progress.make

# Include the compile flags for this target's objects.
include internal/ceres/CMakeFiles/is_close_test.dir/flags.make

internal/ceres/CMakeFiles/is_close_test.dir/is_close_test.cc.o: internal/ceres/CMakeFiles/is_close_test.dir/flags.make
internal/ceres/CMakeFiles/is_close_test.dir/is_close_test.cc.o: /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/internal/ceres/is_close_test.cc
internal/ceres/CMakeFiles/is_close_test.dir/is_close_test.cc.o: internal/ceres/CMakeFiles/is_close_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object internal/ceres/CMakeFiles/is_close_test.dir/is_close_test.cc.o"
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/internal/ceres && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT internal/ceres/CMakeFiles/is_close_test.dir/is_close_test.cc.o -MF CMakeFiles/is_close_test.dir/is_close_test.cc.o.d -o CMakeFiles/is_close_test.dir/is_close_test.cc.o -c /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/internal/ceres/is_close_test.cc

internal/ceres/CMakeFiles/is_close_test.dir/is_close_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/is_close_test.dir/is_close_test.cc.i"
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/internal/ceres && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/internal/ceres/is_close_test.cc > CMakeFiles/is_close_test.dir/is_close_test.cc.i

internal/ceres/CMakeFiles/is_close_test.dir/is_close_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/is_close_test.dir/is_close_test.cc.s"
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/internal/ceres && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/internal/ceres/is_close_test.cc -o CMakeFiles/is_close_test.dir/is_close_test.cc.s

# Object files for target is_close_test
is_close_test_OBJECTS = \
"CMakeFiles/is_close_test.dir/is_close_test.cc.o"

# External object files for target is_close_test
is_close_test_EXTERNAL_OBJECTS =

bin/is_close_test: internal/ceres/CMakeFiles/is_close_test.dir/is_close_test.cc.o
bin/is_close_test: internal/ceres/CMakeFiles/is_close_test.dir/build.make
bin/is_close_test: lib/libtest_util.a
bin/is_close_test: lib/libceres.a
bin/is_close_test: lib/libgtest.a
bin/is_close_test: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/is_close_test: /usr/lib/x86_64-linux-gnu/libblas.so
bin/is_close_test: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/is_close_test: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/is_close_test: /usr/lib/x86_64-linux-gnu/libglog.so.0.4.0
bin/is_close_test: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
bin/is_close_test: /usr/lib/x86_64-linux-gnu/libunwind.so
bin/is_close_test: internal/ceres/CMakeFiles/is_close_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/is_close_test"
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/internal/ceres && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/is_close_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
internal/ceres/CMakeFiles/is_close_test.dir/build: bin/is_close_test
.PHONY : internal/ceres/CMakeFiles/is_close_test.dir/build

internal/ceres/CMakeFiles/is_close_test.dir/clean:
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/internal/ceres && $(CMAKE_COMMAND) -P CMakeFiles/is_close_test.dir/cmake_clean.cmake
.PHONY : internal/ceres/CMakeFiles/is_close_test.dir/clean

internal/ceres/CMakeFiles/is_close_test.dir/depend:
	cd /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0 /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-solver-2.0.0/internal/ceres /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/internal/ceres /home/smkim/spatialai_tutorial/SLAM/Backend/NonLinearOptimization/ceres/ceres-bin/internal/ceres/CMakeFiles/is_close_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : internal/ceres/CMakeFiles/is_close_test.dir/depend

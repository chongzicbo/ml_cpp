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
CMAKE_SOURCE_DIR = /data/bocheng/dev/mylearn/cplus/ml_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /data/bocheng/dev/mylearn/cplus/ml_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/kmeans.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/kmeans.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/kmeans.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kmeans.dir/flags.make

CMakeFiles/kmeans.dir/src/main.cpp.o: CMakeFiles/kmeans.dir/flags.make
CMakeFiles/kmeans.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/kmeans.dir/src/main.cpp.o: CMakeFiles/kmeans.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/bocheng/dev/mylearn/cplus/ml_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kmeans.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kmeans.dir/src/main.cpp.o -MF CMakeFiles/kmeans.dir/src/main.cpp.o.d -o CMakeFiles/kmeans.dir/src/main.cpp.o -c /data/bocheng/dev/mylearn/cplus/ml_cpp/src/main.cpp

CMakeFiles/kmeans.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/bocheng/dev/mylearn/cplus/ml_cpp/src/main.cpp > CMakeFiles/kmeans.dir/src/main.cpp.i

CMakeFiles/kmeans.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/bocheng/dev/mylearn/cplus/ml_cpp/src/main.cpp -o CMakeFiles/kmeans.dir/src/main.cpp.s

CMakeFiles/kmeans.dir/src/kmeans.cpp.o: CMakeFiles/kmeans.dir/flags.make
CMakeFiles/kmeans.dir/src/kmeans.cpp.o: ../src/kmeans.cpp
CMakeFiles/kmeans.dir/src/kmeans.cpp.o: CMakeFiles/kmeans.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/bocheng/dev/mylearn/cplus/ml_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/kmeans.dir/src/kmeans.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kmeans.dir/src/kmeans.cpp.o -MF CMakeFiles/kmeans.dir/src/kmeans.cpp.o.d -o CMakeFiles/kmeans.dir/src/kmeans.cpp.o -c /data/bocheng/dev/mylearn/cplus/ml_cpp/src/kmeans.cpp

CMakeFiles/kmeans.dir/src/kmeans.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans.dir/src/kmeans.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/bocheng/dev/mylearn/cplus/ml_cpp/src/kmeans.cpp > CMakeFiles/kmeans.dir/src/kmeans.cpp.i

CMakeFiles/kmeans.dir/src/kmeans.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans.dir/src/kmeans.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/bocheng/dev/mylearn/cplus/ml_cpp/src/kmeans.cpp -o CMakeFiles/kmeans.dir/src/kmeans.cpp.s

CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o: CMakeFiles/kmeans.dir/flags.make
CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o: ../src/gradient_descent.cpp
CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o: CMakeFiles/kmeans.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/data/bocheng/dev/mylearn/cplus/ml_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o -MF CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o.d -o CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o -c /data/bocheng/dev/mylearn/cplus/ml_cpp/src/gradient_descent.cpp

CMakeFiles/kmeans.dir/src/gradient_descent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans.dir/src/gradient_descent.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /data/bocheng/dev/mylearn/cplus/ml_cpp/src/gradient_descent.cpp > CMakeFiles/kmeans.dir/src/gradient_descent.cpp.i

CMakeFiles/kmeans.dir/src/gradient_descent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans.dir/src/gradient_descent.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /data/bocheng/dev/mylearn/cplus/ml_cpp/src/gradient_descent.cpp -o CMakeFiles/kmeans.dir/src/gradient_descent.cpp.s

# Object files for target kmeans
kmeans_OBJECTS = \
"CMakeFiles/kmeans.dir/src/main.cpp.o" \
"CMakeFiles/kmeans.dir/src/kmeans.cpp.o" \
"CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o"

# External object files for target kmeans
kmeans_EXTERNAL_OBJECTS =

kmeans: CMakeFiles/kmeans.dir/src/main.cpp.o
kmeans: CMakeFiles/kmeans.dir/src/kmeans.cpp.o
kmeans: CMakeFiles/kmeans.dir/src/gradient_descent.cpp.o
kmeans: CMakeFiles/kmeans.dir/build.make
kmeans: CMakeFiles/kmeans.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/data/bocheng/dev/mylearn/cplus/ml_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable kmeans"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmeans.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kmeans.dir/build: kmeans
.PHONY : CMakeFiles/kmeans.dir/build

CMakeFiles/kmeans.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kmeans.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kmeans.dir/clean

CMakeFiles/kmeans.dir/depend:
	cd /data/bocheng/dev/mylearn/cplus/ml_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /data/bocheng/dev/mylearn/cplus/ml_cpp /data/bocheng/dev/mylearn/cplus/ml_cpp /data/bocheng/dev/mylearn/cplus/ml_cpp/build /data/bocheng/dev/mylearn/cplus/ml_cpp/build /data/bocheng/dev/mylearn/cplus/ml_cpp/build/CMakeFiles/kmeans.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kmeans.dir/depend


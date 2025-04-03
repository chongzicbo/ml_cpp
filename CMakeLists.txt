cmake_minimum_required(VERSION 3.10)
project(ml_cpp)

set(CMAKE_CXX_STANDARD 17)


# Create directories for source and header files
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add source files
add_executable(attention
        src/main.cpp
        src/kmeans.cpp
        src/gradient_descent.cpp
        src/attention.cpp
        src/self_attention.cpp
        src/autodiff.cpp
        src/eigen_self_attention.cpp
        )

# Add include directories
target_include_directories(attention PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen
        )

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
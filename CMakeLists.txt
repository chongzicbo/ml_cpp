cmake_minimum_required(VERSION 3.10)
project(ml_cpp)

set(CMAKE_CXX_STANDARD 17)

# 添加可执行文件
add_executable(ml src/main.cpp src/kmeans.cpp src/gradient_descent.cpp src/autodiff.cpp src/attention.cpp include/self_attention.hpp src/self_attention.cpp)

# 包含头文件目录
target_include_directories(ml PRIVATE ${PROJECT_SOURCE_DIR}/include)
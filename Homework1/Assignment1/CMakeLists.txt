cmake_minimum_required(VERSION 3.10)
project(Rasterizer)
set(CMAKE_CXX_STANDARD 17)

set(OpenCV_DIR D:/zzn/Lib/opencv/build/MinGW64/install)

find_package(OpenCV REQUIRED)

include_directories(D:/zzn/Lib/Eigen3/include)

add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp Triangle.hpp Triangle.cpp)
target_link_libraries(Rasterizer ${OpenCV_LIBRARIES})

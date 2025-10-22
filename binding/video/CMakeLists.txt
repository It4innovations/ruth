cmake_minimum_required(VERSION 3.26)
project(video)

set(CMAKE_CXX_STANDARD 20)

# HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# OpenMP
find_package(OpenMP REQUIRED)

# Add executables
add_executable(video_preprocess main_omp.cpp)

# Include HDF5 headers
target_include_directories(video_preprocess PRIVATE ${HDF5_INCLUDE_DIRS})

# Link HDF5 + OpenMP libraries
target_link_libraries(video_preprocess PRIVATE ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX)

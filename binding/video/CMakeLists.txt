cmake_minimum_required(VERSION 3.26)
project(video)

set(CMAKE_CXX_STANDARD 20)

# HDF5
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# OpenMP
find_package(OpenMP REQUIRED)

# Source files for video_preprocess
set(VIDEO_PREPROCESS_SOURCES
    main.cpp
)

set(VIDEO_PREPROCESS_HEADERS
    common_types.hpp
    stage1_loading.hpp
    stage2_aggregation.hpp
    stage3_output.hpp
    pipeline.hpp
)

# Add executables
add_executable(video_preprocess ${VIDEO_PREPROCESS_SOURCES} ${VIDEO_PREPROCESS_HEADERS})
add_executable(merge_h5 merge_h5.cpp)
add_executable(test_aggregation test_aggregation.cpp ${VIDEO_PREPROCESS_HEADERS})

# Include HDF5 headers
target_include_directories(video_preprocess PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(merge_h5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(test_aggregation PRIVATE ${HDF5_INCLUDE_DIRS})

# Link HDF5 + OpenMP libraries
target_link_libraries(video_preprocess PRIVATE ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX)
target_link_libraries(merge_h5 PRIVATE ${HDF5_LIBRARIES})
target_link_libraries(test_aggregation PRIVATE ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX)

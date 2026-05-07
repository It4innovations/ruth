cmake_minimum_required(VERSION 3.10)
project(ruth)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING 
        "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

# Set the possible values of build type for cmake-gui
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS 
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-invalid-offsetof")
#set(PYTHON_EXECUTABLE "/bin/python")
add_definitions(-DNDEBUG)

# --- Fetch pybind11 from GitHub ---
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.12.0   # or latest stable tag
)
fetchContent_MakeAvailable(pybind11)

find_package(MPI REQUIRED)

# Add the directories for the libraries
add_subdirectory(RoutingAlg)
add_subdirectory(ace)

add_library(ruth src/lib.cpp)

# Link libraries to your main program
target_link_libraries(ruth PUBLIC routing inih ace MPI::MPI_CXX pthread)
target_include_directories(ruth PUBLIC RoutingAlg)
target_include_directories(ruth PUBLIC RoutingAlg/include)
target_include_directories(ruth PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(ruth PUBLIC src)

pybind11_add_module(ruthlib src/bindings.cpp)
target_link_libraries(ruthlib PRIVATE routing inih ace MPI::MPI_CXX pthread)
target_include_directories(ruthlib PRIVATE RoutingAlg)
target_include_directories(ruthlib PRIVATE RoutingAlg/include)
target_include_directories(ruthlib PRIVATE ${HDF5_INCLUDE_DIRS})

# Add executable target from main.cpp
# add_executable(ruth_exec src/main.cpp)
# target_link_libraries(ruth_exec PRIVATE ruth)
# target_include_directories(ruth_exec PRIVATE src)
# target_include_directories(ruth_exec PRIVATE ace/src)

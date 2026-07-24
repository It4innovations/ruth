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
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Add the directories for the libraries
add_subdirectory(RoutingAlg)
add_subdirectory(ace)

set(ADVANCE_SOURCES
    src/advance_state.cpp
    src/advance_kernel.cpp
    src/advance_workloads.cpp
    src/advance_orchestration.cpp)

# In Debug / RelWithDebInfo builds, sort the gathered FCDs into a canonical
# (vehicle, time) order so the FCD history is byte-reproducible while debugging.
# Release skips this time-consuming per-step sort (results are identical either
# way — only row order in the output file changes).
set(ADVANCE_SORT_FCDS OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(ADVANCE_SORT_FCDS ON)
endif()

add_library(ruth src/lib.cpp ${ADVANCE_SOURCES})
if(ADVANCE_SORT_FCDS)
  target_compile_definitions(ruth PRIVATE RUTH_ADVANCE_SORT_FCDS)
endif()

# Link libraries to your main program
target_link_libraries(ruth PUBLIC routing inih ace MPI::MPI_CXX pthread)
target_include_directories(ruth PUBLIC RoutingAlg)
target_include_directories(ruth PUBLIC RoutingAlg/include)
target_include_directories(ruth PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(ruth PUBLIC src)

pybind11_add_module(ruthlib src/bindings.cpp ${ADVANCE_SOURCES})
if(ADVANCE_SORT_FCDS)
  target_compile_definitions(ruthlib PRIVATE RUTH_ADVANCE_SORT_FCDS)
endif()
target_link_libraries(ruthlib PRIVATE routing inih ace MPI::MPI_CXX pthread)
target_include_directories(ruthlib PRIVATE RoutingAlg)
target_include_directories(ruthlib PRIVATE RoutingAlg/include)
target_include_directories(ruthlib PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(ruthlib PRIVATE src)

# GlobalView C++ module with spatial indexing
pybind11_add_module(globalview_cpp src/globalview_bindings.cpp)
target_include_directories(globalview_cpp PRIVATE src)
target_link_libraries(globalview_cpp PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(globalview_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)


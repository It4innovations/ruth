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

# In Debug / RelWithDebInfo builds, check the ordering invariant
# GlobalView::get_segment_speed relies on: a vehicle's records never go backwards
# in time in storage order, so keeping the last one keeps the most recent one.
# Nothing in the type system enforces that chain, so it is verified at the point
# of use and aborts loudly. Release compiles the check out entirely.
#
# NOTE: this must reach globalview_cpp as well, not just ruth/ruthlib --
# get_segment_speed is called from Python through that module (see
# globalview_wrapper.take_segment_speeds), so an unflagged globalview_cpp would
# leave the production call path unchecked.
set(GV_CHECK_FCD_ORDER OFF)
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
  set(GV_CHECK_FCD_ORDER ON)
endif()

add_library(ruth src/lib.cpp ${ADVANCE_SOURCES})
if(GV_CHECK_FCD_ORDER)
  target_compile_definitions(ruth PRIVATE RUTH_GV_CHECK_FCD_ORDER)
endif()

# Link libraries to your main program
target_link_libraries(ruth PUBLIC routing inih ace MPI::MPI_CXX pthread)
target_include_directories(ruth PUBLIC RoutingAlg)
target_include_directories(ruth PUBLIC RoutingAlg/include)
target_include_directories(ruth PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(ruth PUBLIC src)

pybind11_add_module(ruthlib src/bindings.cpp ${ADVANCE_SOURCES})
if(GV_CHECK_FCD_ORDER)
  target_compile_definitions(ruthlib PRIVATE RUTH_GV_CHECK_FCD_ORDER)
endif()
target_link_libraries(ruthlib PRIVATE routing inih ace MPI::MPI_CXX pthread)
target_include_directories(ruthlib PRIVATE RoutingAlg)
target_include_directories(ruthlib PRIVATE RoutingAlg/include)
target_include_directories(ruthlib PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(ruthlib PRIVATE src)

# GlobalView C++ module with spatial indexing
pybind11_add_module(globalview_cpp src/globalview_bindings.cpp)
if(GV_CHECK_FCD_ORDER)
  target_compile_definitions(globalview_cpp PRIVATE RUTH_GV_CHECK_FCD_ORDER)
endif()
target_include_directories(globalview_cpp PRIVATE src)
target_link_libraries(globalview_cpp PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(globalview_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)


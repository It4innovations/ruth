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

# Reproducing route.py's arithmetic exactly, or running the fastest kernel.
#
# route.py evaluates the per-step move as two statements and therefore two
# roundings:
#     elapsed_m    = frequency_s * speed_mps
#     end_position = start_position + elapsed_m
# advance_kernel.cpp writes the same thing as one expression. Wherever FMA is
# available -- any -march that implies it, which the Cray wrappers set from the
# node -- GCC and Clang contract the pair into a single fused multiply-add that
# rounds once, and the result sits up to 1 ULP from Python's. Measured at 4-7%
# of moving vehicles per step for 5-10 s frequencies.
#
# Contraction is the ONLY -O3 transform permitted to change a floating point
# result without -ffast-math (reassociation and vectorised reductions are not),
# so turning it off is enough: optimisation stays at the build type's level and
# nothing else about the build changes.
#
#   ON  (default) -- positions match the serial Python run bit for bit.
#   OFF           -- the fused kernel, up to 1 ULP from serial per move.
#
# Switch with -DRUTH_REPRODUCIBLE_FP=OFF at configure time; no source changes.
option(RUTH_REPRODUCIBLE_FP
    "Match route.py's floating point exactly (disables FMA contraction)" ON)

if(RUTH_REPRODUCIBLE_FP)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-ffp-contract=off" HAVE_FFP_CONTRACT_OFF)
    if(NOT HAVE_FFP_CONTRACT_OFF)
        message(WARNING
            "RUTH_REPRODUCIBLE_FP is ON but this compiler rejects "
            "-ffp-contract=off. Vehicle positions may differ from the serial "
            "run by 1 ULP per move.")
    endif()
endif()

set(ADVANCE_SOURCES
    src/advance_state.cpp
    src/advance_kernel.cpp
    src/advance_workloads.cpp
    src/advance_orchestration.cpp
    src/advance_store.cpp
    src/advance_queues.cpp)

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
# This target links ace, so a fatal check in globalview.hpp can tear the job down
# through CommInterface::abort() instead of aborting one rank. NOT set for
# globalview_cpp, which must stay free of MPI (see the note in globalview.hpp).
target_compile_definitions(ruth PRIVATE RUTH_WITH_ACE)

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
# This target links ace, so a fatal check in globalview.hpp can tear the job down
# through CommInterface::abort() instead of aborting one rank. NOT set for
# globalview_cpp, which must stay free of MPI (see the note in globalview.hpp).
target_compile_definitions(ruthlib PRIVATE RUTH_WITH_ACE)
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

# Applied after the targets exist. globalview_cpp is included because the serial
# path reaches the LoS through it and the distributed path through ruthlib's
# copy of the same header -- compiling them differently would stop them being
# the same function.
if(RUTH_REPRODUCIBLE_FP AND HAVE_FFP_CONTRACT_OFF)
    foreach(_ruth_fp_target ruth ruthlib globalview_cpp)
        target_compile_options(${_ruth_fp_target} PRIVATE -ffp-contract=off)
    endforeach()
endif()

# Printed because the whole point is to A/B the two builds: this line says which
# one produced the trajectory digests in the log beside it.
message(STATUS "ruth: RUTH_REPRODUCIBLE_FP=${RUTH_REPRODUCIBLE_FP} "
               "(ON = positions match serial Python bit for bit)")


cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;80;89;90"
      CACHE STRING "CUDA architectures (Volta, Ampere, Ada, Hopper)")
endif()

project(tenryu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

option(TENRYU_ENABLE_MPI "Enable MPI support" OFF)
option(TENRYU_ENABLE_HDF5 "Enable HDF5 support" ON)
option(TENRYU_ENABLE_PYTHON "Enable Python support" ON)
# NVTX macro plumbing exists for optional instrumentation, but v1.0 does not
# yet wire all profiling regions to this switch.
option(TENRYU_ENABLE_NVTX "Enable NVTX support" OFF)
option(TENRYU_ENABLE_HYPRE "Enable Hypre support" OFF)
find_package(amgx QUIET)
option(TENRYU_ENABLE_AMGX "Enable AmgX support for 2D FLD solves" ${amgx_FOUND})
set(TENRYU_RFA_V2_MODE "FULL" CACHE STRING
    "Radial Fourier audit v2 mode: OFF, STUB, DUMMY_BUFFER, or FULL")
set_property(CACHE TENRYU_RFA_V2_MODE PROPERTY STRINGS OFF STUB DUMMY_BUFFER FULL)
set(_TENRYU_RFA_V2_MODE_VALUES OFF STUB DUMMY_BUFFER FULL)
if(NOT TENRYU_RFA_V2_MODE IN_LIST _TENRYU_RFA_V2_MODE_VALUES)
  message(FATAL_ERROR
      "TENRYU_RFA_V2_MODE must be one of OFF, STUB, DUMMY_BUFFER, FULL")
endif()
set(TENRYU_GPU_AWARE_MPI_COMPILE 0)

add_compile_definitions(
  TENRYU_ENABLE_MPI=$<BOOL:${TENRYU_ENABLE_MPI}>
  TENRYU_ENABLE_HDF5=$<BOOL:${TENRYU_ENABLE_HDF5}>
  TENRYU_ENABLE_PYTHON=$<BOOL:${TENRYU_ENABLE_PYTHON}>
  TENRYU_ENABLE_NVTX=$<BOOL:${TENRYU_ENABLE_NVTX}>
  TENRYU_ENABLE_HYPRE=$<BOOL:${TENRYU_ENABLE_HYPRE}>
  TENRYU_ENABLE_AMGX=$<BOOL:${TENRYU_ENABLE_AMGX}>
  TENRYU_HAVE_AMGX=$<BOOL:${amgx_FOUND}>
  TENRYU_RFA_V2_MODE_${TENRYU_RFA_V2_MODE}=1
)

include(cmake/Dependencies.cmake)
include(cmake/ConfigAbiDeps.cmake)

add_compile_definitions(
  TENRYU_GPU_AWARE_MPI_COMPILE=${TENRYU_GPU_AWARE_MPI_COMPILE}
)

include(CTest)

add_subdirectory(src)

# Snapshot exports (beta) may omit these trees.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
  add_subdirectory(benchmarks)
endif()

if(BUILD_TESTING AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_subdirectory(tests)
endif()

# 3.27 needed for $<COMPILE_ONLY:...> (test link rewiring, tests/CMakeLists.txt).
cmake_minimum_required(VERSION 3.27)

# Default CUDA architectures. When the caller specifies nothing (neither
# -DCMAKE_CUDA_ARCHITECTURES nor the CUDAARCHS environment variable), probe
# the local GPUs with nvidia-smi and build for exactly the detected compute
# capabilities. Hosts without a visible GPU (cluster login nodes, packaging
# machines) fall back to the portable multi-arch set. Rationale: the test
# suite links ~550 RDC executables, and every extra architecture multiplies
# every embedded GPU code object, so the portable set costs ~4x the disk.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(_tenryu_cuda_archs "70;80;89;90")
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    RESULT_VARIABLE _tenryu_smi_result
    OUTPUT_VARIABLE _tenryu_smi_output
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_tenryu_smi_result EQUAL 0)
    string(REPLACE "\n" ";" _tenryu_smi_output "${_tenryu_smi_output}")
    set(_tenryu_detected_archs "")
    foreach(_tenryu_cap IN LISTS _tenryu_smi_output)
      string(STRIP "${_tenryu_cap}" _tenryu_cap)
      string(REPLACE "." "" _tenryu_cap "${_tenryu_cap}")
      if(_tenryu_cap MATCHES "^[0-9]+$")
        list(APPEND _tenryu_detected_archs "${_tenryu_cap}")
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _tenryu_detected_archs)
    if(NOT "${_tenryu_detected_archs}" STREQUAL "")
      set(_tenryu_cuda_archs "${_tenryu_detected_archs}")
    endif()
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${_tenryu_cuda_archs}"
      CACHE STRING "CUDA architectures (detected local GPU, or portable Volta/Ampere/Ada/Hopper set)")
  unset(_tenryu_cuda_archs)
endif()

project(tenryu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Compress every embedded GPU code object (SASS/PTX fatbins). Lossless: the
# CUDA runtime decompresses at module load, executed device code is
# byte-identical. This applies to the relocatable per-TU fatbins as well,
# which dominate RDC-linked binary size.
string(APPEND CMAKE_CUDA_FLAGS " -Xfatbin=-compress-all")

# All library objects are linked both into the static `tenryu` binary and
# into the shared `tenryu_lib` used by the test executables, so they must be
# position independent. (Host codegen effect is nil in practice: Ubuntu
# toolchains already default executables to PIE.)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

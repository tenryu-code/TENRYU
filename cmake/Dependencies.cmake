include(FetchContent)

set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "Use external fmt in spdlog" FORCE)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(CUDAToolkit REQUIRED)

if(TENRYU_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  include(CheckGPUAwareMPI)
endif()

if(TENRYU_ENABLE_HDF5)
  # Auto-detect HDF5 install root for Debian/Ubuntu multiarch layouts
  if(NOT DEFINED HDF5_ROOT AND NOT DEFINED ENV{HDF5_ROOT})
    set(_tenryu_hdf5_candidates
        "/usr/lib/x86_64-linux-gnu/hdf5/openmpi"
        "/usr/lib/x86_64-linux-gnu/hdf5/mpich"
        "/usr/lib/x86_64-linux-gnu/hdf5/serial"
        "/opt/hdf5"
    )
    foreach(_cand IN LISTS _tenryu_hdf5_candidates)
      if(EXISTS "${_cand}/lib/libhdf5.so" OR EXISTS "${_cand}/lib/libhdf5.so.1")
        set(HDF5_ROOT "${_cand}" CACHE PATH "HDF5 install root (auto-detected)")
        message(STATUS "TENRYU: HDF5 auto-detected at ${HDF5_ROOT}")
        break()
      endif()
    endforeach()
    unset(_tenryu_hdf5_candidates)
    unset(_cand)
  endif()
  find_package(HDF5 REQUIRED COMPONENTS C CXX)
  if(HDF5_IS_PARALLEL)
    message(STATUS "Parallel HDF5 detected; MPI is required (Ubuntu: install libopenmpi-dev).")
    find_package(MPI REQUIRED COMPONENTS CXX)
  endif()
endif()

if(TENRYU_ENABLE_HYPRE)
  find_package(Hypre REQUIRED)
endif()

if(TENRYU_ENABLE_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use FindPython in pybind11" FORCE)

  # Query the selected Python for pybind11's cmake dir (portable fallback).
  if(NOT pybind11_DIR AND Python3_EXECUTABLE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE TENRYU_PYBIND11_CMAKE_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()

  find_package(pybind11 CONFIG QUIET
               HINTS
               "${pybind11_DIR}"
               "${TENRYU_PYBIND11_CMAKE_DIR}")
endif()

find_package(Catch2 3 QUIET)
find_package(spdlog QUIET)
find_package(CLI11 QUIET)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.0
)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.12.0
)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
)

if(TENRYU_ENABLE_PYTHON)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v3.0.1
  )
endif()


if(NOT Catch2_FOUND)
  FetchContent_MakeAvailable(Catch2)
endif()

if(NOT spdlog_FOUND)
  FetchContent_MakeAvailable(spdlog)
endif()

if(NOT CLI11_FOUND)
  if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/CLI11/CMakeLists.txt")
    set(FETCHCONTENT_SOURCE_DIR_CLI11 "${PROJECT_SOURCE_DIR}/third_party/CLI11")
  endif()
  FetchContent_MakeAvailable(CLI11)
endif()

if(TENRYU_ENABLE_PYTHON AND NOT pybind11_FOUND)
  FetchContent_MakeAvailable(pybind11)
endif()

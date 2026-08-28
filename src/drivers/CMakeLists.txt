add_library(tenryu_drivers STATIC
  cli.cpp
  cmd_run.cpp
  cmd_validate.cpp
  cmd_freeze.cpp
  cmd_verify.cpp
  verify_conduction_eigenmode_1d.cpp
  verify_snb_1d.cpp
  verify_snb_2d.cpp
  verify_conduction_pattle_1d.cpp
  verify_sn_cylindrical.cpp
  verify_sn_lathrop.cpp
  verify_fld_1d_mg_marshak.cpp
  verify_braginskii.cpp
  verify_braginskii_2d.cpp
  verify_button_morph_2d.cpp
  verify_rad_gamma.cpp
)

target_include_directories(tenryu_drivers PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_drivers PUBLIC
  tenryu_coupling
  tenryu_io
  tenryu_diagnostics
  tenryu_verification
  tenryu_materials
  tenryu_core
  tenryu_parallel
  CLI11::CLI11
)

if(TENRYU_ENABLE_PYTHON)
  target_link_libraries(tenryu_drivers PRIVATE pybind11::embed)
endif()

set_target_properties(tenryu_drivers PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

if(TENRYU_ENABLE_PYTHON)
  add_library(tenryu_namelist_embed STATIC
    ${PROJECT_SOURCE_DIR}/src/core/namelist/api.cpp
  )
  target_include_directories(tenryu_namelist_embed PUBLIC ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(tenryu_namelist_embed PUBLIC
    tenryu_core
    pybind11::embed
  )
  # No per-archive device link: CMake 3.28 Ninja emits static-library device-link archive deps as order-only (measured 2026-07-28, tmp/rdc_forensics_20260728.log), which goes stale when a linked archive's device fatbin hash changes; the tenryu executable device link carries these archives as REAL deps and resolves all device symbols, so this per-library device link is redundant and disabled.
  set_target_properties(tenryu_namelist_embed PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
endif()

add_executable(tenryu main.cpp)

target_link_libraries(tenryu PRIVATE tenryu_drivers)

if(TENRYU_ENABLE_PYTHON)
  target_link_libraries(tenryu PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,tenryu_namelist_embed>"
  )
endif()

set_target_properties(tenryu PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# The shared library below exists solely for the test executables; snapshot
# exports without a tests/ tree skip it (same condition as the top-level
# add_subdirectory(tests) gate).
if(BUILD_TESTING AND EXISTS ${PROJECT_SOURCE_DIR}/tests)

  # ---------------------------------------------------------------------------
  # tenryu_lib: one shared library carrying the whole TENRYU code base with a
  # single device link. Test executables link this instead of re-device-linking
  # every static archive (which costs ~36 MB of NVIDIA code sections per test
  # executable, ~20 GB per build tree across ~550 tests). The production
  # `tenryu` binary above intentionally keeps its fully static link.
  #
  # - Whole-archive linking: every object participates, so kernel/host
  #   registration objects cannot be dropped by archive member selection.
  # - The plain archive entries bring the transitive third-party link
  #   closure (HDF5, cusparse, spdlog, CLI11, ...).
  # - LINKER:--no-undefined makes any missing dependency a loud link error
  #   instead of a runtime failure inside a test.
  # - tenryu_namelist_embed is included too (core's namelist runtime calls
  #   init_embedded_module()). Tests that link it explicitly as a whole
  #   archive are excluded from the rewiring and keep their full static
  #   link, so no process carries both copies.
  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/tenryu_lib_anchor.cu)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/tenryu_lib_anchor.cu
      "// Anchor translation unit: tenryu_lib gets its code from the archives.\n")
  endif()

  add_library(tenryu_lib SHARED
    ${CMAKE_CURRENT_BINARY_DIR}/tenryu_lib_anchor.cu
  )

  set(TENRYU_LIB_ARCHIVES
    tenryu_drivers tenryu_coupling tenryu_verification tenryu_diagnostics
    tenryu_io tenryu_materials tenryu_hydro tenryu_radiation tenryu_laser
    tenryu_burn tenryu_mesh tenryu_parallel tenryu_core)

  if(TENRYU_ENABLE_PYTHON)
    list(APPEND TENRYU_LIB_ARCHIVES tenryu_namelist_embed)
  endif()

  # Plain entries provide the CUDA device-link inputs, the target-level build
  # dependencies, and the transitive third-party link closure (HDF5, cusparse,
  # spdlog, CLI11, ...).
  target_link_libraries(tenryu_lib PRIVATE ${TENRYU_LIB_ARCHIVES})

  # CMake's WHOLE_ARCHIVE link feature cannot be used here: these archives
  # PUBLIC-link each other, so each also reaches the link line as a plain
  # transitive item, and CMake rejects one item carrying two link features.
  # Raw linker flags do the same job: this flag group precedes the library
  # list on the link line and forces every archive member in (registration
  # objects cannot be dropped by member selection); the later plain
  # occurrences of the same archives are skipped as already loaded.
  target_link_options(tenryu_lib PRIVATE "LINKER:--whole-archive")
  foreach(_tenryu_lib_archive IN LISTS TENRYU_LIB_ARCHIVES)
    target_link_options(tenryu_lib PRIVATE
      "LINKER:$<TARGET_FILE:${_tenryu_lib_archive}>")
  endforeach()
  target_link_options(tenryu_lib PRIVATE "LINKER:--no-whole-archive")

  if(TENRYU_ENABLE_PYTHON)
    target_link_libraries(tenryu_lib PRIVATE pybind11::embed)
  endif()
  if(TENRYU_ENABLE_MPI)
    target_link_libraries(tenryu_lib PRIVATE MPI::MPI_CXX)
  endif()

  target_link_options(tenryu_lib PRIVATE "LINKER:--no-undefined")

  set_target_properties(tenryu_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared
  )

endif()

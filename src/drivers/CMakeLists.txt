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

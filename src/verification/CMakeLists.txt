add_library(tenryu_verification STATIC
  sedov_analytic.cpp
  noh_analytic.cpp
  diffusion_ref.cpp
  escape_valve_audit.cpp
  tier_threshold.cu
  audit_summary.cpp
  positivity_tracker.cu
  marshak.cu
  laser_analytic.cu
)

target_include_directories(tenryu_verification PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_verification PUBLIC
  tenryu_core
  tenryu_parallel
)

set_target_properties(tenryu_verification PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

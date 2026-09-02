add_library(tenryu_coupling STATIC
  dispatcher_decision.cpp
  dt_controller_check.cu
  persistent_loop.cu
  driver.cpp
  driver_fld_energy.cu
  reale_trigger.cu
  rad_gamma_coupling.cu
  rad_gamma_coupling.cuh
  driver_reclose.cu
  driver_retry_snapshot.cu
  driver_safety_audit.cu
  profile_observability.cpp
  reale_mode.cpp
  staged_dt_gate.cpp
  source_terms.cu
)

target_include_directories(tenryu_coupling PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_coupling PUBLIC
  tenryu_hydro
  tenryu_radiation
  tenryu_laser
  tenryu_burn
  tenryu_diagnostics
  tenryu_materials
  tenryu_mesh
  tenryu_io
  tenryu_core
  tenryu_parallel
  tenryu_verification
)

set_target_properties(tenryu_coupling PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

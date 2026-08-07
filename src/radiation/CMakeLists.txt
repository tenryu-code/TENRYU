add_library(tenryu_radiation STATIC
  imc.cpp
  ddmc.cpp
  holo_selector.cpp
  holo_lo_solver.cpp
  sn_transport_1d.cpp
  sn_cyl_quadrature_1d.cpp
  sn_transport_gpu.cu
  sn_transport_1d_gpu.cu
  sn_transport_2d_gpu.cu
  sn_dsa_1d_gpu.cu
  sn_material_newton_gpu.cu
  holo_lo_state.cu
  mode_selector.cpp
  diffusion_conversion.cu
  diffusion_interface.cu
  diffusion_source_solve.cu
  deterministic_diffusion_1d.cu
  ../numerics/rkl2_sts.cu
  ddmc_coefficients.cpp
  ddmc_diffusion_1d.cu
  mmatrix_check.cpp
  ddmc_transport.cpp
  ddmc_transport_2d.cpp
  ddmc_transport_gpu.cu
  ddmc_transport_2d_gpu.cu
  rw_transport_gpu.cu
  fld_1d_gpu.cu
  fld_2d_rz_gpu.cu
  fld_substage_audit_drain.cpp
  amgx_solver.cpp
  ddmc_momentum.cpp
  interface.cu
  group_structure.cu
  groups.cu
  planck_table.cu
  radiation_init.cu
  particle_pool.cu
  particle_reid.cu
  pool_stats.cu
  source.cu
  fleck.cu
  nlte_coeffs.cpp
  nlte_coeffs.cu
  composite_sort.cu
  census_comb.cu
  difference_residualization.cu
  tally.cu
  rad_lite_mesh.cu
  imc_transport_persistent.cu
  imc_transport_2d.cu
  energy_budget.cu
)

target_include_directories(tenryu_radiation PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_radiation PUBLIC
  tenryu_mesh
  tenryu_materials
  tenryu_core
  tenryu_parallel
  CUDA::cusparse
)

if(TENRYU_ENABLE_AMGX AND amgx_FOUND)
  if(TARGET amgx::amgx)
    target_link_libraries(tenryu_radiation PUBLIC amgx::amgx)
  elseif(TARGET amgx)
    target_link_libraries(tenryu_radiation PUBLIC amgx)
  endif()
endif()

set_target_properties(tenryu_radiation PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

add_library(tenryu_radiation STATIC
  radiation_step.cpp
  sn_cyl_quadrature_1d.cpp
  sn_transport_gpu.cu
  sn_transport_1d_gpu.cu
  sn_transport_2d_gpu.cu
  sn_dsa_1d_gpu.cu
  sn_ld_1d_gpu.cu
  sn_material_newton_gpu.cu
  fld_1d_gpu.cu
  material_electron_eos_1d.cu
  fld_2d_rz_gpu.cu
  fld_substage_audit_drain.cpp
  amgx_solver.cpp
  group_structure.cu
  groups.cu
  planck_table.cu
  radiation_init.cu
  nlte_coeffs.cu
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

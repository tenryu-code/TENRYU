add_library(tenryu_hydro STATIC
  hydro.cpp
  adaptive_av_gate.cu
  anti_hourglass.cu
  anti_hourglass.cuh
  ale_align_monitor.cpp
  ale_align_monitor.hpp
  ale_align_rezone.cpp
  ale_align_rezone.hpp
  ale_axis_band_controller.cu
  axis_ale_rezone.cu
  boundary_pressure_force.cu
  boundary_pressure_force.cuh
  axis_band_margin.cu
  axis_band_remap.cu
  axis_band_guard.cu
  artificial_viscosity.cu
  braginskii_viscosity.cu
  braginskii_viscosity_2d.cu
  braginskii_viscosity.cuh
  compatible_av_csw.cu
  compatible_av_csw.cuh
  compatible_force_work_2d.cu
  compatible_force_work_2d.cuh
  compatible_subzonal_pressure.cu
  compatible_subzonal_pressure.cuh
  conservation_audit.cu
  conservation_audit.hpp
  central_pseudo_core.cu
  central_pseudo_core.cuh
  central_core_1d_submodel.cpp
  central_core_1d_submodel.hpp
  pole_angular_derefine.cu
  pole_angular_derefine.cuh
  core_freeze_ale.cu
  core_freeze_ale.cuh
  boundary.cu
  boundary_2d.cu
  button_morph_ale.cu
  button_morph_ale.hpp
  corner_jacobian_quality.cu
  corner_j_predict_cfl.cu
  corner_j_predict_cfl.hpp
  ring7_seam_quotient_remap.cu
  cfl.cu
  conduction.cu
  conduction_snb_1d.cu
  conduction_snb_1d.cuh
  conduction_snb_2d.cu
  cd_local_rezone.cu
  eos_context.cu
  euler_window_blend.cu
  ale_gcl.cu
  ale_identity_diag.cu
  ale_driver.cu
  ale_remap_2d_rz.cu
  corner_mass_remap_audit.cu
  ke_fixup_deposit.cu
  local_rezone.cu
  ale_1d_driver.cu
  ale_1d_diagnostics.cu
  ale_1d_sensor.cu
  ale_1d_rezone.cu
  ale_1d_remap.cu
  ale_1d_velocity_project.cu
  kershaw_geometry.cuh
  kershaw_stencil.cuh
  kershaw_boundary.cuh
  ale_1d_types.cuh
  ale_1d_driver.cuh
  ale_1d_diagnostics.cuh
  ale_1d_sensor.cuh
  ale_1d_rezone.cuh
  ale_1d_remap.cuh
  ale_1d_velocity_project.cuh
  axis_ale_rezone.cuh
  ale_rezone.cuh
  ale_scaled_reference.cuh
  ale_mode.hpp
  cd_local_rezone.cuh
  local_rezone.cuh
  ale_remap.cuh
  ale_remap_2d_rz.hpp
  ale_velocity_project.cuh
  ale_driver.cuh
  hypre_solver.cuh
  hllc_z_flux_2d_rz.cu
  hllc_z_flux_2d_rz.hpp
  hydro_1d.cu
  hydro_2d.cu
  mesh_regime.cu
  hydro_multiblock_topology.cuh
  per_material_eos_project.cu
  per_material_eos_accessors.cuh
  per_material_eos_project.cuh
  pole_axis_constraints.cuh
  pole_axis_bbsw.hpp
  pole_angular_coarsen.cu
  pole_angular_coarsen.cuh
  plic_fast_path.cu
  plic_geometry.cu
  plic_normal.cu
  plic_remap.cu
  reference_barrier_ale.cu
  reference_barrier_ale.hpp
  rz_corner_mass.cuh
  rz_corner_kinetic.cuh
  ring7_seam_quotient_remap.cuh
  rz_geometric_cfl.cu
  shock_tracker.cu
  state_supply_bc.cu
)

target_include_directories(tenryu_hydro PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_hydro PUBLIC
  tenryu_mesh
  tenryu_materials
  tenryu_diagnostics
  tenryu_core
  tenryu_parallel
  CUDA::cusparse
)

if(TENRYU_ENABLE_HYPRE)
  target_link_libraries(tenryu_hydro PUBLIC Hypre::Hypre)
endif()

set_target_properties(tenryu_hydro PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

tenryu_attach_config_state_abi_deps(tenryu_hydro)

target_sources(tenryu_hydro PRIVATE
  rollback_guard.cu
)

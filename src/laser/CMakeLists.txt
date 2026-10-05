add_library(tenryu_laser STATIC
  coordinate_transform.cu
  laser_map_1d_gpu.cu
  laser_mesh.cu
  bilinear_interpolation.cu
  waveform_table.cu
  beams.cu
  refraction.cu
  ib_absorption.cu
  ray_init.cu
  ray_init_1d_gpu.cu
  fast_trace_1d.cu
  ray_trace.cu
  raytrace_skip.cu
  deposit_transfer.cu
  hot_e_eta_model.cpp
  port_geometry.cpp
  port_section_chi.cpp
  port_section_chi.cu
  port_section_overlap.cpp
  port_section_s1_gpu.cu
  port_section_s1_host.cpp
  sector_adapter.cpp
  sector_phase_space.cpp
  hot_electron_1d.cpp
  hot_electron_1d_gpu.cu
  laser.cu
  cbet.cu
  cbet_stage_gpu.cu
  cbet_stage_host.cpp
  deposit_1d_gpu.cu
  hot_e_inputs_gpu.cu
  hot_e_transport_1d_gpu.cu
  hot_electron_1d.cpp
  hot_electron_2d.cpp
  hot_electron_2d_gpu.cu
  coordinate_transform.cuh
  laser_map_1d_gpu.cuh
  laser_mesh.cuh
  laser_mesh_bodies.cuh
  bilinear_interpolation.cuh
  waveform_table.cuh
  beams.cuh
  refraction.cuh
  ib_absorption.cuh
  ray_init.cuh
  ray_init_1d_gpu.cuh
  fast_trace_1d.cuh
  ray_trace.cuh
  ray_trace_bodies.cuh
  raytrace_skip.cuh
  deposit_transfer.cuh
  port_section_chi_lookup.cuh
  port_section_s1_gpu.cuh
  port_section_s1_host.hpp
  hot_electron_1d.cuh
  hot_electron_1d_gpu.cuh
  laser.cuh
  cbet.cuh
  cbet_stage_gpu.cuh
  cbet_stage_host.hpp
  deposit_1d_gpu.cuh
  hot_e_inputs_gpu.cuh
  hot_e_transport_1d_gpu.cuh
  hot_electron_1d.cuh
  hot_electron_2d.cuh
  hot_electron_2d_gpu.cuh
)

target_include_directories(tenryu_laser PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_laser PUBLIC
  tenryu_mesh
  tenryu_materials
  tenryu_core
  tenryu_parallel
)

target_compile_options(tenryu_laser PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(tenryu_laser PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# The device build of the port_section phase-space table, the device CBET stages, the device
# hot-electron model inputs, the device 1D laser map and deposit redistribution, the device 1D ray
# rings and the device 1D hot-electron transport repeat their host references' arithmetic
# operation by operation (and the deposit's double-double sums need exact two-sum steps);
# contraction into fused multiply-adds would change its rounding.
set_source_files_properties(port_section_s1_gpu.cu cbet_stage_gpu.cu hot_e_inputs_gpu.cu
                            laser_map_1d_gpu.cu deposit_1d_gpu.cu ray_init_1d_gpu.cu
                            hot_e_transport_1d_gpu.cu
                            PROPERTIES COMPILE_OPTIONS "-fmad=false")

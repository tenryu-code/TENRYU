add_library(tenryu_laser STATIC
  coordinate_transform.cu
  laser_mesh.cu
  bilinear_interpolation.cu
  waveform_table.cu
  beams.cu
  refraction.cu
  ib_absorption.cu
  ray_init.cu
  fast_trace_1d.cu
  ray_trace.cu
  raytrace_skip.cu
  deposit_transfer.cu
  hot_e_eta_model.cpp
  port_geometry.cpp
  port_section_chi.cpp
  port_section_chi.cu
  port_section_overlap.cpp
  sector_adapter.cpp
  sector_phase_space.cpp
  hot_electron_1d.cpp
  hot_electron_1d_gpu.cu
  laser.cu
  cbet.cu
  hot_electron_1d.cpp
  hot_electron_2d.cpp
  hot_electron_2d_gpu.cu
  coordinate_transform.cuh
  laser_mesh.cuh
  laser_mesh_bodies.cuh
  bilinear_interpolation.cuh
  waveform_table.cuh
  beams.cuh
  refraction.cuh
  ib_absorption.cuh
  ray_init.cuh
  fast_trace_1d.cuh
  ray_trace.cuh
  ray_trace_bodies.cuh
  raytrace_skip.cuh
  deposit_transfer.cuh
  hot_electron_1d.cuh
  hot_electron_1d_gpu.cuh
  laser.cuh
  cbet.cuh
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

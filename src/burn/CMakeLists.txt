add_library(tenryu_burn STATIC
  burn_stage.cpp
  burn_stage.hpp
  burn_stage_gpu.cu
  burn_stage_gpu.cuh
  burn_stage_2d.cpp
  burn_stage_2d.hpp
  neutron_heating.cpp
  neutron_heating.hpp
  neutron_heating_device.cuh
  neutron_moments.cpp
  neutron_moments.hpp
  corman_diffusion.cu
  corman_diffusion.cuh
  corman_diffusion_2d.cu
  corman_diffusion_2d.cuh
  mc_transport.cu
  mc_transport.cuh
  screening.cpp
  screening.hpp
  screening_device.cuh
  network_gpu.cu
  burn_constants.hpp
  deposition.cuh
  reactivity.cuh
  network.cuh
  network_gpu.cuh
  partition.hpp
  partition.cpp
  partition_device.cuh
)

target_include_directories(tenryu_burn PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_burn PUBLIC tenryu_core tenryu_parallel
                                          CUDA::cusparse)

target_compile_options(tenryu_burn PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(tenryu_burn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

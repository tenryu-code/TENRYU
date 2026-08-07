add_library(tenryu_parallel STATIC
  partition.cpp
  halo_exchange.cu
  comm_buffers.cpp
  reduction.cpp
  particle_migration.cu
)

target_include_directories(tenryu_parallel PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_parallel PUBLIC
  tenryu_core
  tenryu_mesh
)
target_link_libraries(tenryu_parallel PRIVATE CUDA::cudart)

if(TENRYU_ENABLE_MPI)
  target_link_libraries(tenryu_parallel PUBLIC MPI::MPI_CXX)
endif()

set_target_properties(tenryu_parallel PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

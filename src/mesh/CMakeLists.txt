add_library(tenryu_mesh STATIC
  boundary_loop_constraint.cpp
  candidate_mesh_admissibility.cu
  cone_shell.cu
  mesh.cu
  pentagon_belt_shell.cu
)

target_include_directories(tenryu_mesh PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(tenryu_mesh PUBLIC tenryu_core)

set_target_properties(tenryu_mesh PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

target_sources(tenryu_mesh PRIVATE
  polygon_mesh.cpp
)

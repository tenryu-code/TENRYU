add_library(tenryu_mesh STATIC
  boundary_carrier.cpp
  boundary_loop_constraint.cpp
  candidate_mesh_admissibility.cu
  carrier_coverage.cpp
  cone_shell.cu
  generator_targets.cu
  generator_targets.cuh
  mesh.cu
  pentagon_belt_shell.cu
  poly_geom.cu
  poly_geom.cuh
  reconnect.cu
  reconnect.cuh
  reference_flat_path.cpp
  reference_flat_path.hpp
  ring_pages.cpp
  tessellation/cecr_cavity.cpp
  tessellation/exact_construct.cpp
  tessellation/gpu_classify.cu
  tessellation/gpu_circumcenter.cu
  tessellation/delaunay_2d.cpp
  tessellation/delaunay_audit.cpp
  tessellation/voronoi_vertex_classes.cpp
  tessellation/voronoi_dual.cpp
  tessellation/domain_restrict.cpp
  tessellation/boundary_arrangement.cpp
  tessellation/weld_quotient.cpp
  tessellation/restricted_dcel.cpp
  tessellation/canonical_serialize.cpp
  tessellation/lloyd_skip.cpp
  tessellation/legacy_adapter.cpp
  voronoi_rz.cu
  z_reflection.cpp
)

target_include_directories(tenryu_mesh PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(tenryu_mesh PUBLIC tenryu_core)

set_target_properties(tenryu_mesh PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

set_source_files_properties(
  tessellation/gpu_classify.cu
  tessellation/gpu_circumcenter.cu
  PROPERTIES COMPILE_OPTIONS "-fmad=false")

target_sources(tenryu_mesh PRIVATE
  polygon_mesh.cpp
)

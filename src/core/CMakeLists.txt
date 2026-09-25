add_library(tenryu_core STATIC
  version.cpp
  config.cpp
  device_scratch.cu
  device_pack.cu
  deterministic_sum.cu
  mesh_transaction.cu
  error.cpp
  state.cpp
  state_per_material_init.cpp
  auto_zone.cpp
  zoning_intent.cpp
  mesh_requirement.cpp
  ${PROJECT_SOURCE_DIR}/src/materials/ionmix_reader.cpp
  ${PROJECT_SOURCE_DIR}/src/materials/zbar_tf.cpp
  namelist/frozen_table.cpp
  namelist/laser_zeff_resolution.cpp
  namelist/geometry_eval.cpp
  namelist/geometry_eval_volume_cut.cpp
)

set_source_files_properties(namelist/geometry_eval_volume_cut.cpp
  PROPERTIES LANGUAGE CUDA
)

if(TENRYU_ENABLE_PYTHON)
  target_sources(tenryu_core PRIVATE
    namelist/builder.cpp
    namelist/runtime.cpp
    namelist/freeze.cpp
  )
endif()

target_include_directories(tenryu_core PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_core PUBLIC spdlog::spdlog)
# PUBLIC: the core headers include the CUDA runtime and Thrust, so every target
# that includes them (plain C++ translation units too) needs the toolkit's
# include directories. With PRIVATE the host compiler found them only where the
# toolkit sits on its default search path (a standard /usr/local/cuda install
# does not: src/mesh/z_reflection.cpp failed with thrust/device_vector.h not
# found). A static library's private link dependencies already reach the final
# link, so the linked libraries are unchanged.
target_link_libraries(tenryu_core PUBLIC CUDA::cudart)

if(TENRYU_ENABLE_PYTHON)
  target_link_libraries(tenryu_core PRIVATE
    pybind11::embed
  )
endif()

if(TENRYU_ENABLE_MPI)
  target_link_libraries(tenryu_core PRIVATE MPI::MPI_CXX)
endif()

set_target_properties(tenryu_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

tenryu_attach_config_state_abi_deps(tenryu_core)

target_sources(tenryu_core PRIVATE
  field_measure.cpp
)

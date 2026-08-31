add_library(tenryu_core STATIC
  version.cpp
  config.cpp
  device_scratch.cu
  device_pack.cu
  mesh_transaction.cu
  error.cpp
  state.cpp
  state_per_material_init.cpp
  auto_zone.cpp
  zoning_intent.cpp
  ${PROJECT_SOURCE_DIR}/src/materials/ionmix_reader.cpp
  ${PROJECT_SOURCE_DIR}/src/materials/zbar_tf.cpp
  namelist/frozen_table.cpp
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
target_link_libraries(tenryu_core PRIVATE CUDA::cudart)

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

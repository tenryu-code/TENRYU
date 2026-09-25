add_library(tenryu_io STATIC
  hdf5_writer.cpp
  hdf5_reader.cpp
  output_manager.cpp
)

target_include_directories(tenryu_io PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_io PUBLIC
  tenryu_core
  tenryu_diagnostics
  tenryu_hydro
  tenryu_parallel
)

if(TENRYU_ENABLE_HDF5)
  if(TARGET HDF5::HDF5)
    target_link_libraries(tenryu_io PUBLIC HDF5::HDF5)
  else()
    target_include_directories(tenryu_io PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(tenryu_io PUBLIC ${HDF5_LIBRARIES})
  endif()
  if(HDF5_IS_PARALLEL)
    target_link_libraries(tenryu_io PUBLIC MPI::MPI_CXX)
  endif()
  # zlib compresses the large deflate datasets of snapshots and checkpoints in
  # parallel (hdf5_writer.cpp, the deflate filter's own compress2 call).
  find_package(ZLIB REQUIRED)
  target_link_libraries(tenryu_io PRIVATE ZLIB::ZLIB)
endif()

set_target_properties(tenryu_io PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

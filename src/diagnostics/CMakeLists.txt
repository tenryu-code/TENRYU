add_library(tenryu_diagnostics STATIC
  diagnostics.cu
  history_shape_diag_device.cu
  energy_budget.cu
  operator_energy_residuals.cu
  mesh_deform_attribution.cu
  mesh_degeneracy_forensics.cu
  mesh_diag_dump.cpp
  escape_valve_history.cpp
  positivity_history.cpp
  radial_fourier_audit.cu
  history_writer.cpp
)

set_source_files_properties(history_shape_diag_device.cu
  PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>"
)

target_include_directories(tenryu_diagnostics PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_diagnostics PUBLIC
  tenryu_core
  tenryu_laser
  tenryu_verification
)

if(TENRYU_ENABLE_HDF5)
  if(TARGET HDF5::HDF5)
    target_link_libraries(tenryu_diagnostics PUBLIC HDF5::HDF5)
  else()
    target_include_directories(tenryu_diagnostics PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(tenryu_diagnostics PUBLIC ${HDF5_LIBRARIES})
  endif()
  if(HDF5_IS_PARALLEL)
    target_link_libraries(tenryu_diagnostics PUBLIC MPI::MPI_CXX)
  endif()
endif()

set_target_properties(tenryu_diagnostics PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

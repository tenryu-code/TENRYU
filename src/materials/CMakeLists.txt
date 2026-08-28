add_library(tenryu_materials STATIC
  eos.cpp
  eos_rho_e_table.cpp
  eos_table.cpp
  helmholtz_jet.cpp
  helmholtz_jet_device.cu
  helmholtz_bspline.cpp
  helmholtz_spline.cpp
  helmholtz_spline_device.cu
  mie_gruneisen.cpp
  mie_gruneisen_device.cu
  opacity.cu
  opacity_diagnostics.cu
  eos_rho_e_device.cu
  eos_device_table.cu
  opacity_table.cpp
  # zbar_tf.cpp is compiled into tenryu_core (materials links it PUBLIC);
  # listing it here too would duplicate its symbols under whole-archive
  # linking (tenryu_lib).
  sesame_reader.cpp
  tmat_reader.cpp
  zmoment_state.cu
)

target_include_directories(tenryu_materials PUBLIC ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(tenryu_materials PUBLIC
  tenryu_core
)

if(TENRYU_ENABLE_HDF5)
  if(TARGET HDF5::HDF5)
    target_link_libraries(tenryu_materials PUBLIC HDF5::HDF5)
  else()
    target_include_directories(tenryu_materials PUBLIC ${HDF5_INCLUDE_DIRS})
    target_link_libraries(tenryu_materials PUBLIC ${HDF5_LIBRARIES})
  endif()
  if(HDF5_IS_PARALLEL)
    target_link_libraries(tenryu_materials PUBLIC MPI::MPI_CXX)
  endif()
endif()

set_target_properties(tenryu_materials PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

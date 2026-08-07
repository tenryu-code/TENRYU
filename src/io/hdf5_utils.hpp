#pragma once

#include <hdf5.h>

namespace tenryu::io {

/// Wrapper around H5Lexists that suppresses HDF5 diagnostic output.
/// HDF5 1.10.x prints error stacks to stderr when intermediate path
/// components do not exist, even though H5Lexists is designed to return
/// false in that case. This wrapper temporarily disables the automatic
/// error printing so that expected "not found" results remain silent.
inline htri_t h5_link_exists(hid_t loc, const char* name, hid_t lapl = H5P_DEFAULT) {
  H5E_auto2_t old_func = nullptr;
  void* old_client_data = nullptr;
  H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  const htri_t result = H5Lexists(loc, name, lapl);
  H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
  return result;
}

}  // namespace tenryu::io

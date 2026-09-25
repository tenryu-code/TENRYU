#pragma once

#include <mutex>

namespace tenryu::core {

// Serializes the HDF5 calls made during the time loop. The history writer
// writes its row batches on a worker thread while snapshots, checkpoints and
// the other history files are written from the main thread, and the HDF5
// library may be built without thread safety. Recursive: a holder may reach
// another locked entry point on the same thread.
inline std::recursive_mutex& hdf5_mutex() {
  static std::recursive_mutex mutex;
  return mutex;
}

}  // namespace tenryu::core

#pragma once

#include <bitset>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>

#include "core/config.hpp"
#include "core/namelist/errors.hpp"

#if TENRYU_ENABLE_PYTHON
#include <pybind11/pybind11.h>
#endif

namespace tenryu::core::namelist {

#if TENRYU_ENABLE_PYTHON
namespace py = pybind11;
#endif

struct PythonCallable {
  std::string name;
  std::string repr;
  std::string source_hash = "unavailable";
  bool detected = false;
};

using NamelistConfig = tenryu::core::Config;

#if TENRYU_ENABLE_PYTHON
class Builder {
 public:
  enum class Block : std::size_t {
    Main = 0,
    Mesh,
    Materials,
    Geometry,
    Radiation,
    Laser,
    Numerics,
    Output,
    Diagnostics,
    Parallel,
    Burn,
    Count
  };

  static constexpr std::size_t kBlockCount = static_cast<std::size_t>(Block::Count);

  NamelistConfig config;
  std::map<std::string, PythonCallable> callables;
  std::map<std::string, py::object> callable_objects;
  std::bitset<kBlockCount> blocks_called;
  bool main_name_explicit = false;
  bool motion_explicitly_set = false;
  bool laser_rays_per_beam_explicit = false;
  bool mesh_r_min_explicit = false;
  bool mesh_r_max_explicit = false;
  bool mesh_z_min_explicit = false;
  bool mesh_z_max_explicit = false;
  double hydro_t_start_eV = 0.0;

  void set_main(py::dict kwargs);
  void set_mesh(py::dict kwargs);
  void set_materials(py::dict kwargs);
  void set_geometry(py::dict kwargs);
  void set_radiation(py::dict kwargs);
  void set_laser(py::dict kwargs);
  void set_numerics(py::dict kwargs);
  void set_output(py::dict kwargs);
  void set_diagnostics(py::dict kwargs);
  void set_parallel(py::dict kwargs);
  void set_burn(py::dict kwargs);

  void validate();

 private:
  void mark_block_called(Block block);
  void register_callable(const std::string& path,
                         const PythonCallable& callable,
                         py::handle callable_obj);
};
#else
class Builder {};
#endif

void begin_build(Builder& builder);
void end_build();
Builder* active_builder();
Builder& require_active_builder();

}  // namespace tenryu::core::namelist

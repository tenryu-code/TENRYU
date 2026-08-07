#include "core/namelist/api.hpp"

#if TENRYU_ENABLE_PYTHON

#include <regex>
#include <string>

#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace tenryu::core::namelist {
namespace {

std::string normalize_repr(std::string repr) {
  static const std::regex kAddressRegex{"0x[0-9a-fA-F]+"};
  return std::regex_replace(std::move(repr), kAddressRegex, "<addr>");
}

std::string sha256_string(const std::string& text) {
  try {
    py::object digest = py::module_::import("hashlib").attr("sha256")(py::bytes(text));
    return "sha256:" + py::str(digest.attr("hexdigest")()).cast<std::string>();
  } catch (...) {
    return "unavailable";
  }
}

std::string py_type_name(const py::handle value) {
  if (value.is_none()) {
    return "NoneType";
  }
  return py::str(py::type::of(value).attr("__name__")).cast<std::string>();
}

bool py_callable(const py::handle value) {
  return PyCallable_Check(value.ptr()) != 0;
}

py::dict kwargs_to_dict(const py::kwargs& kwargs) {
  py::dict out;
  for (const auto item : kwargs) {
    out[item.first] = item.second;
  }
  return out;
}

}  // namespace

int _tenryu_namelist_module_anchor = 1;

bool is_callable(py::handle value) {
  return py_callable(value);
}

PythonCallable extract_callable_info(py::handle value) {
  if (!py::isinstance<py::function>(value)) {
    if (py_callable(value)) {
      throw ConfigError("callable must be a function (def/lambda), got " +
                        py_type_name(value));
    }
    throw ValueError("value must be callable, got " + py_type_name(value));
  }

  PythonCallable info;
  info.detected = true;
  try {
    info.name = py::str(value.attr("__name__")).cast<std::string>();
  } catch (...) {
    info.name = "<anonymous>";
  }
  info.repr = normalize_repr(py::repr(value).cast<std::string>());
  try {
    py::object source_obj = py::module_::import("inspect").attr("getsource")(value);
    info.source_hash = sha256_string(py::str(source_obj).cast<std::string>());
  } catch (...) {
    info.source_hash = "unavailable";
  }
  return info;
}

void init_embedded_module() {
  (void)_tenryu_namelist_module_anchor;
  if (!Py_IsInitialized()) {
    return;
  }
  try {
    py::module_::import("tenryu_namelist");
  } catch (const py::error_already_set& e) {
    throw ConfigError(std::string("embedded module not linked: ") + e.what());
  }
}

PYBIND11_EMBEDDED_MODULE(tenryu_namelist, m) {
  m.doc() = "TENRYU Python namelist API";

  m.def("Main", [](py::kwargs kwargs) {
    require_active_builder().set_main(kwargs_to_dict(kwargs));
  });
  m.def("Mesh", [](py::kwargs kwargs) {
    require_active_builder().set_mesh(kwargs_to_dict(kwargs));
  });
  m.def("Materials", [](py::kwargs kwargs) {
    require_active_builder().set_materials(kwargs_to_dict(kwargs));
  });
  m.def("Geometry", [](py::kwargs kwargs) {
    require_active_builder().set_geometry(kwargs_to_dict(kwargs));
  });
  m.def("Radiation", [](py::kwargs kwargs) {
    require_active_builder().set_radiation(kwargs_to_dict(kwargs));
  });
  m.def("Laser", [](py::kwargs kwargs) {
    require_active_builder().set_laser(kwargs_to_dict(kwargs));
  });
  m.def("Numerics", [](py::kwargs kwargs) {
    require_active_builder().set_numerics(kwargs_to_dict(kwargs));
  });
  m.def("Output", [](py::kwargs kwargs) {
    require_active_builder().set_output(kwargs_to_dict(kwargs));
  });
  m.def("Diagnostics", [](py::kwargs kwargs) {
    require_active_builder().set_diagnostics(kwargs_to_dict(kwargs));
  });
  m.def("Parallel", [](py::kwargs kwargs) {
    require_active_builder().set_parallel(kwargs_to_dict(kwargs));
  });
  m.def("Burn", [](py::kwargs kwargs) {
    require_active_builder().set_burn(kwargs_to_dict(kwargs));
  });

  m.def("Material", [](py::kwargs kwargs) { return kwargs_to_dict(kwargs); });
  m.def("LaserBeam", [](py::kwargs kwargs) { return kwargs_to_dict(kwargs); });
}

}  // namespace tenryu::core::namelist

#else

namespace tenryu::core::namelist {

int _tenryu_namelist_module_anchor = 1;

void init_embedded_module() {
  (void)_tenryu_namelist_module_anchor;
}

}  // namespace tenryu::core::namelist

#endif

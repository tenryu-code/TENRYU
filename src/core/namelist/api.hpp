#pragma once

#include "core/namelist/builder.hpp"

#if TENRYU_ENABLE_PYTHON
#include <pybind11/pybind11.h>
#endif

namespace tenryu::core::namelist {

#if TENRYU_ENABLE_PYTHON
namespace py = pybind11;

bool is_callable(py::handle value);
PythonCallable extract_callable_info(py::handle value);
#endif

void init_embedded_module();
extern int _tenryu_namelist_module_anchor;

}  // namespace tenryu::core::namelist

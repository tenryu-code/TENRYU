#include "core/namelist/runtime.hpp"

#if TENRYU_ENABLE_PYTHON

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/eval.h>
#include <pybind11/embed.h>

#include "core/namelist/api.hpp"
#include "core/namelist/freeze.hpp"

namespace tenryu::core::namelist {
namespace {

namespace py = pybind11;

class BuildScope final {
 public:
  explicit BuildScope(Builder& builder) {
    begin_build(builder);
  }
  ~BuildScope() {
    end_build();
  }
};

std::filesystem::path expand_tilde(std::string path) {
  if (path.empty() || path[0] != '~') {
    return std::filesystem::path(std::move(path));
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr || std::string_view(home).empty()) {
    throw ConfigError("Cannot expand '~': HOME is not set");
  }
  if (path.size() == 1) {
    return std::filesystem::path(home);
  }
  if (path[1] != '/') {
    throw ConfigError("Only '~/...' tilde expansion is supported");
  }
  return std::filesystem::path(home) / path.substr(2);
}

std::filesystem::path normalize_path(const std::string& user_path) {
  auto path = expand_tilde(user_path);
  path = std::filesystem::absolute(path).lexically_normal();
  return path;
}

std::string load_text_file(const std::filesystem::path& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw ConfigError("Failed to open namelist file: " + path.string());
  }
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

std::string sha256_string(const std::string& text) {
  try {
    py::object digest = py::module_::import("hashlib").attr("sha256")(py::bytes(text));
    return "sha256:" + py::str(digest.attr("hexdigest")()).cast<std::string>();
  } catch (...) {
    return "unavailable";
  }
}

void add_parent_dir_to_sys_path(const std::filesystem::path& parent_dir) {
  py::list sys_path = py::module_::import("sys").attr("path");
  const std::string parent = parent_dir.string();
  for (const py::handle entry : sys_path) {
    if (py::str(entry).cast<std::string>() == parent) {
      return;
    }
  }
  sys_path.insert(0, parent);
}

}  // namespace

PythonGuard::PythonGuard() {
  if (!Py_IsInitialized()) {
    py::initialize_interpreter();
    owns_interpreter_ = true;
  }
}

PythonGuard::~PythonGuard() {
  if (owns_interpreter_ && Py_IsInitialized()) {
    py::finalize_interpreter();
  }
}

const NamelistConfig& Runtime::execute(const std::string& namelist_path) {
  builder_ = Builder{};
  BuildScope build_scope(builder_);

  resolved_namelist_path_ = normalize_path(namelist_path);
  if (!std::filesystem::exists(resolved_namelist_path_)) {
    throw ConfigError("Namelist file does not exist: " + resolved_namelist_path_.string());
  }
  if (!std::filesystem::is_regular_file(resolved_namelist_path_)) {
    throw ConfigError("Namelist path is not a regular file: " +
                      resolved_namelist_path_.string());
  }

  builder_.config.meta.namelist_source_path = resolved_namelist_path_.string();
  builder_.config.meta.namelist_source_dir =
      std::filesystem::current_path().string();
  builder_.config.meta.namelist_source_hash =
      sha256_string(load_text_file(resolved_namelist_path_));

  try {
    init_embedded_module();
    add_parent_dir_to_sys_path(resolved_namelist_path_.parent_path());

    py::dict globals;
    globals["__builtins__"] = py::module_::import("builtins");
    globals["__name__"] = "__main__";
    globals["__file__"] = resolved_namelist_path_.string();

    py::exec("from tenryu_namelist import *", globals, globals);
    py::eval_file(resolved_namelist_path_.string(), globals, globals);
  } catch (const py::error_already_set& e) {
    throw ConfigError(std::string("Python execution failed: ") + e.what());
  }

  builder_.validate();
  builder_.config.meta.frozen_config_json =
      Freeze::to_checkpoint_json(builder_.config);
  return builder_.config;
}

const Builder& Runtime::builder() const {
  return builder_;
}

const NamelistConfig& Runtime::config() const {
  return builder_.config;
}

const std::filesystem::path& Runtime::resolved_namelist_path() const {
  return resolved_namelist_path_;
}

}  // namespace tenryu::core::namelist

#else

namespace tenryu::core::namelist {

PythonGuard::PythonGuard() = default;
PythonGuard::~PythonGuard() = default;

const NamelistConfig& Runtime::execute(const std::string&) {
  throw ConfigError("Python support is disabled (TENRYU_ENABLE_PYTHON=OFF)");
}

const Builder& Runtime::builder() const {
  return builder_;
}

const NamelistConfig& Runtime::config() const {
  return builder_.config;
}

const std::filesystem::path& Runtime::resolved_namelist_path() const {
  return resolved_namelist_path_;
}

}  // namespace tenryu::core::namelist

#endif

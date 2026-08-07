#pragma once

#include <filesystem>
#include <string>

#include "core/namelist/builder.hpp"

namespace tenryu::core::namelist {

class PythonGuard {
 public:
  PythonGuard();
  ~PythonGuard();

  PythonGuard(const PythonGuard&) = delete;
  PythonGuard& operator=(const PythonGuard&) = delete;

 private:
  bool owns_interpreter_ = false;
};

class Runtime {
 public:
  Runtime() = default;

  const NamelistConfig& execute(const std::string& namelist_path);

  const Builder& builder() const;
  const NamelistConfig& config() const;
  const std::filesystem::path& resolved_namelist_path() const;

 private:
  Builder builder_;
  std::filesystem::path resolved_namelist_path_;
};

}  // namespace tenryu::core::namelist

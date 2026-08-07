#pragma once

#include <cstddef>
#include <vector>

namespace tenryu::materials {

struct EOSTable;

class EOSRhoETable {
 public:
  struct Result {
    double P = 0.0;
    double T = 0.0;
    double cv = 0.0;
    double cs2 = 0.0;
  };

  void build_from_rhoT_table(const EOSTable& rhoT_table, bool use_linear_grid = false);

  [[nodiscard]] Result eval(double rho, double e) const;

  [[nodiscard]] bool empty() const noexcept {
    return n_rho_ == 0 || n_e_ == 0;
  }

  [[nodiscard]] std::size_t n_rho() const noexcept {
    return static_cast<std::size_t>(n_rho_);
  }

  [[nodiscard]] std::size_t n_e() const noexcept {
    return static_cast<std::size_t>(n_e_);
  }

  [[nodiscard]] std::size_t flat_index(std::size_t i_rho,
                                       std::size_t j_e) const noexcept {
    return j_e * n_rho() + i_rho;
  }

  [[nodiscard]] bool use_log_grid() const noexcept {
    return use_log_grid_;
  }

  [[nodiscard]] const std::vector<double>& log_rho_grid() const noexcept {
    return log_rho_grid_;
  }

  [[nodiscard]] const std::vector<double>& log_e_grid() const noexcept {
    return log_e_grid_;
  }

  [[nodiscard]] const std::vector<double>& P_table() const noexcept {
    return P_table_;
  }

  [[nodiscard]] const std::vector<double>& T_table() const noexcept {
    return T_table_;
  }

  [[nodiscard]] const std::vector<double>& P_xx() const noexcept {
    return P_xx_;
  }

  [[nodiscard]] const std::vector<double>& P_yy() const noexcept {
    return P_yy_;
  }

  [[nodiscard]] const std::vector<double>& P_xxyy() const noexcept {
    return P_xxyy_;
  }

  [[nodiscard]] const std::vector<double>& T_xx() const noexcept {
    return T_xx_;
  }

  [[nodiscard]] const std::vector<double>& T_yy() const noexcept {
    return T_yy_;
  }

  [[nodiscard]] const std::vector<double>& T_xxyy() const noexcept {
    return T_xxyy_;
  }

 private:
  std::vector<double> log_rho_grid_;
  std::vector<double> log_e_grid_;
  std::vector<double> P_table_;
  std::vector<double> T_table_;
  std::vector<double> P_xx_;
  std::vector<double> P_yy_;
  std::vector<double> P_xxyy_;
  std::vector<double> T_xx_;
  std::vector<double> T_yy_;
  std::vector<double> T_xxyy_;
  bool use_log_grid_ = true;
  int n_rho_ = 0;
  int n_e_ = 0;
};

}  // namespace tenryu::materials

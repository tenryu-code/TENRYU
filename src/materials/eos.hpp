#pragma once

namespace tenryu::materials {

class EOS {
 public:
  virtual ~EOS() = default;

  virtual double pressure(double rho, double e) const = 0;
  virtual double energy(double rho, double T) const = 0;
  virtual double temperature(double rho, double e) const = 0;
  virtual double sound_speed(double rho, double e) const = 0;
  virtual double cv(double rho, double T) const = 0;

  virtual double pressure_ion(double rho, double e_i) const {
    return pressure(rho, e_i);
  }

  virtual double pressure_electron(double, double) const {
    return 0.0;
  }

  virtual double energy_ion(double rho, double T_i) const {
    return energy(rho, T_i);
  }

  virtual double energy_electron(double, double) const {
    return 0.0;
  }
};

class IdealGasEOS final : public EOS {
 public:
  IdealGasEOS(double gamma, double A, double Z_bar);

  double pressure(double rho, double e) const override;
  double energy(double rho, double T) const override;
  double temperature(double rho, double e) const override;
  double sound_speed(double rho, double e) const override;
  double cv(double rho, double T) const override;
  double pressure_ion(double rho, double e_i) const override;
  double pressure_electron(double rho, double e_e) const override;
  double energy_ion(double rho, double T_i) const override;
  double energy_electron(double rho, double T_e) const override;

  [[nodiscard]] double gamma() const noexcept {
    return gamma_;
  }

  [[nodiscard]] double A() const noexcept {
    return A_;
  }

  [[nodiscard]] double Z_bar() const noexcept {
    return Z_bar_;
  }

  [[nodiscard]] double Cv_total() const noexcept {
    return cv_total_;
  }

  [[nodiscard]] double Cv_i() const noexcept {
    return cv_i_;
  }

  [[nodiscard]] double Cv_e() const noexcept {
    return cv_e_;
  }

 private:
  double gamma_ = 5.0 / 3.0;
  double A_ = 1.0;
  double Z_bar_ = 0.0;
  double cv_total_ = 0.0;
  double cv_i_ = 0.0;
  double cv_e_ = 0.0;
};

}  // namespace tenryu::materials

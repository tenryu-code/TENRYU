#pragma once

namespace tenryu::hydro {

class LagrangianStep {
 public:
  LagrangianStep() = default;

  void execute(double dt);
};

}  // namespace tenryu::hydro

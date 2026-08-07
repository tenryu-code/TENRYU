#pragma once

#include <string>

namespace tenryu::radiation {

struct AmgxSolveRequest {
  int n_rows = 0;
  int nnz = 0;
  const int* row_offsets = nullptr;
  const int* col_indices = nullptr;
  const double* values = nullptr;
  const double* rhs = nullptr;
  double* solution = nullptr;
  std::string preset = "AGGREGATION_JACOBI";
};

bool amgx_solver_available();

void solve_with_amgx(const AmgxSolveRequest& request);

}  // namespace tenryu::radiation

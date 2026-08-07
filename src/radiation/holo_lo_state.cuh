#pragma once

namespace tenryu::radiation {

void initialize_holo_lo_state_cuda(double* E_lo,
                                   int n_cells,
                                   int n_groups);

void initialize_holo_lo_from_lte_cuda(double* E_lo,
                                      const double* Te,
                                      int n_cells,
                                      int n_groups);

}  // namespace tenryu::radiation

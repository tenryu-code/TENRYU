#pragma once

namespace tenryu::core {
struct State;
}

namespace tenryu::hydro {

enum class CornerCollapseLedgerTrial {
  Accepted,
  RejectedPostrestore,
};

void corner_collapse_ledger_capture(const core::State& state,
                                    int step,
                                    double t_n,
                                    double dt,
                                    CornerCollapseLedgerTrial trial);

void corner_collapse_ledger_flush();

}  // namespace tenryu::hydro

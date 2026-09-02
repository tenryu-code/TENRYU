#pragma once

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::hydro {

enum class EntropyLedgerStage {
  Hydro = 0,
  Radiation = 1,
  Ale = 2,
};

void entropy_ledger_step_begin(core::State& state, const core::Config& cfg);

void entropy_ledger_begin(core::State& state,
                          const core::Config& cfg,
                          EntropyLedgerStage stage);

void entropy_ledger_end(core::State& state,
                        const core::Config& cfg,
                        EntropyLedgerStage stage);

void entropy_ledger_accumulate_work_buckets(core::State& state,
                                            const core::Config& cfg,
                                            double dt);

void entropy_ledger_log(core::State& state, const core::Config& cfg);

}  // namespace tenryu::hydro

// Extract (a reference copy, not a compilable unit): the parts of the driver and of State that served only
// the retired 1D ALE, as they were at 6d62bf929.
//
//   - src/coupling/driver.cpp: the includes, the conservative_advection assertion that excluded the 1D ALE,
//     and the call between hydro steps (inside Driver::run; the branch was the first of the chain whose
//     next branch is the 2D single-block conservative remap). The call sat between the entropy ledger's
//     ALE stage marks; an applied rezone set State::ale_rezoned, counted State::ale_rezone_invocations and
//     invalidated the per-cell material properties.
//   - src/core/state.hpp / state.cpp: State::ale1d_floor_cooldown_remaining and its two initializations
//     (State::allocate and State::reset).

// ---- src/coupling/driver.cpp, lines 58-59 at 6d62bf929 ----
#include "hydro/ale_1d_driver.cuh"
#include "hydro/ale_1d_types.cuh"

// ---- src/coupling/driver.cpp, lines 8442-8447 at 6d62bf929 ----
    TENRYU_ASSERT(!rad_cell_advection_on ||
                      (is_1d && cfg.mesh.motion == "lagrangian" &&
                       !cfg.numerics.ale1d.enabled &&
                       cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion &&
                       !rad_gamma::gamma_r_43_enabled_from_env()),
                  "conservative_advection requires 1D Lagrangian FLD and no gamma override");

// ---- src/coupling/driver.cpp, lines 13839-13876 at 6d62bf929 ----
    if (is_1d && cfg.numerics.hydro.enabled && cfg.numerics.ale1d.enabled) {
      // 1D V3 ALE is a conservative remap between hydro steps. Unlike the
      // pure Lagrangian 1D hydro update, state.mass may change after this call.
      hydro::entropy_ledger_begin(
          state, cfg, hydro::EntropyLedgerStage::Ale);
      const auto ale1d_out = hydro::ale1d::apply_ale_1d(state, cfg, &eos_ctx);
      hydro::entropy_ledger_end(
          state, cfg, hydro::EntropyLedgerStage::Ale);
      static int ale1d_reject_logged = 0;
      if ((ale1d_out.cadence_triggered || ale1d_out.quality_triggered ||
           ale1d_out.floor_triggered) &&
          !ale1d_out.applied && ale1d_reject_logged < 5) {
        std::ostringstream message;
        message << std::scientific << std::setprecision(3)
                << "[ale1d] step=" << state.step << " triggered(cad="
                << (ale1d_out.cadence_triggered ? 1 : 0)
                << " qual=" << (ale1d_out.quality_triggered ? 1 : 0)
                << " floor=" << (ale1d_out.floor_triggered ? 1 : 0)
                << ") NOT applied: reason="
                << hydro::ale1d::to_string(ale1d_out.skip_reason)
                << " remap_rejected=" << (ale1d_out.remap_rejected ? 1 : 0)
                << " candidate_dt_gain=" << ale1d_out.candidate_dt_gain
                << " mass_err=" << ale1d_out.mass_conservation_rel_err
                << " energy_err=" << ale1d_out.energy_conservation_rel_err
                << " radiation_conservation_rel_err="
                << ale1d_out.radiation_conservation_rel_err
                << " kinetic_energy_drift_rel="
                << ale1d_out.kinetic_energy_drift_rel;
        core::log_info(message.str());
        ++ale1d_reject_logged;
      }
      state.ale_rezoned = ale1d_out.applied;
      if (state.ale_rezoned) {
        state.ale_rezone_invocations += 1;
        // Remap moves volFrac between cells; the shared per-cell effective
        // material properties must be rebuilt before the next closure.
        state.invalidate_cell_material_props();
      }

// ---- src/core/state.hpp, lines 1167-1170 at 6d62bf929 ----
  // ALE1D min-width-floor retrigger cooldown: remaining steps for which the
  // floor-trigger evaluation is skipped after a rejected floor-triggered
  // attempt. In-memory only (not checkpointed).
  int ale1d_floor_cooldown_remaining = 0;

// ---- src/core/state.cpp, line 506 at 6d62bf929 ----
  state.ale1d_floor_cooldown_remaining = 0;

// ---- src/core/state.cpp, line 1116 at 6d62bf929 ----
  ale1d_floor_cooldown_remaining = 0;

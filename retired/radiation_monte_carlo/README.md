# retired/radiation_monte_carlo — the retired Monte Carlo radiation

This directory keeps the Monte Carlo radiation transport of TENRYU (`Radiation.mode = "imc_ddmc"`) out of the
build. Nothing here is compiled: there is no CMakeLists.txt, and no file under `src/`, `tests/` or `examples/`
refers to these files.

| | |
|---|---|
| Methods | Implicit Monte Carlo (Fleck–Cummings, implicit capture, census), Discrete Diffusion Monte Carlo (DDMC) and the IMC⇄DDMC interface, the particle random walk (PGRW), the hybrid deterministic diffusion (RKL2 super-time-stepping), HOLO (high-order/low-order acceleration with the low-order diffusion / quasi-diffusion solve and its S_N closure), the difference formulation, the rad-lite mesh, the photon-particle pool, its sorting and census combing, and the photon migration between MPI ranks |
| Frozen | 2026-04-26 for 1D_SPH (FREEZE-1D-RAD); 2026-07-10 in every dimension (`mode="imc_ddmc"` became a ConfigError) |
| Moved out of the build | 2026-09-29, branch `lane/1d-detach-frozen-kernels-20260929` |
| Last commit where it built and its tests ran | **5bc8f6ce3** (see "Test status at 5bc8f6ce3") |

The production radiation is deterministic: flux-limited diffusion (`mode="multigroup_diffusion"`, NUMERICS §6.7)
and discrete ordinates (`mode="sn_transport"`, NUMERICS §6.8), entered from the driver through
`radiation::RadiationStep` (`src/radiation/radiation_step.{hpp,cpp}`). Until 2026-09-29 that entry was
`radiation::IMC::transport_step`, which dispatched the deterministic modes before its own.

## Layout

The tree mirrors the repository layout.

| Path | Content |
|---|---|
| `src/radiation/` | 76 files moved whole (`git log --follow` shows their history): `imc.*`, `imc_transport_*`, `ddmc*`, `holo*`, `fleck.*`, `source.*`, `tally.*`, `particle_*`, `census_comb.*`, `composite_sort.*`, `pool_stats.*`, `rw_transport_gpu.*`, `difference_residualization.*`, `deterministic_diffusion_1d.*`, `diffusion_{conversion,interface,source_solve}.*`, `rad_lite_mesh.*`, `mode_selector.*`, `mmatrix_check.*`, `interface.*`, `boundary.cuh`, `face_geometry_2d.cuh`, `boundary_distance_2d.cuh`, `cell_radiation_coeffs.hpp`, `energy_budget.*`, the host NLTE coefficients `nlte_coeffs.{cpp,hpp}`, the CPU S_N of the HOLO closure `sn_transport_1d.{cpp,hpp}` |
| `src/radiation/nlte_coeffs_monte_carlo.cu`, `sn_transport_gpu_holo_1d.cu` | extracts: the entry of `nlte_coeffs.cu` that only IMC/DDMC called, and the 1D S_N pieces of the HOLO closure cut out of `sn_transport_gpu.{cu,cuh}` |
| `src/core/rng/` | the Philox helpers of the photon particles (`rng_init.cuh`, the CPU reference `philox_cpu.hpp`) |
| `src/core/namelist/monte_carlo_radiation_config.cpp` | extract: the configuration front end (Config members, builder parsing and validation, frozen-config serialization) of the Monte Carlo radiation and of the keys only it read |
| `src/coupling/driver_monte_carlo.cpp` | extract: the functions of `driver.cpp` that served only the Monte Carlo radiation (the inline parts of `Driver::run` are best read in place at 5bc8f6ce3) |
| `src/coupling/source_terms_radiation.cu` | extract: the radiation source injection (matter update from the IMC/DDMC deposition, net electron source smoothing) |
| `src/drivers/cmd_verify_monte_carlo.cpp` | extract: the verify targets `nlte_*` (7), `imc_ddmc_*` (4), `ddmc_*` (3), `mmatrix_fallback`, `void_passthrough`, `gxii_1d_regression` and the helpers only they used |
| `src/io/hdf5_reader_monte_carlo.cpp` | extract: the checkpoint restore of the photon pool, the DDMC mode map and the HOLO state |
| `src/numerics/rkl2_sts.*`, `src/parallel/particle_migration.*`, `src/verification/diffusion_ref.*` | moved whole: the RKL2 stepper of the hybrid diffusion, the photon migration, the diffusion reference of the DDMC gates |
| `tests/` | 34 test executables moved whole (`test_ddmc_*`, `test_holo_*`, `test_fleck*`, `test_tally`, `test_particle_*`, `test_census_comb`, `test_composite_sort`, `test_mode_selector`, `test_interface`, `test_philox_cpu`, `test_driver_retry_imc_abort`, ...) and extracts of the Monte Carlo test cases of files that stayed (`test_source_terms_radiation.cpp`, `test_ale_1d_skeleton_imc.cu`, `test_hdf5_roundtrip_monte_carlo.cpp`, `test_sn_transport_gpu_1d.cu`) |
| `examples/` | the decks of the retired gates and benchmarks: `verification/ddmc_{diffusion,leak_normalization,multigroup}.py`, `verification/mmatrix_fallback.py`, `perf/P1.py`, `P2.py`, `P3.py`, `P5.py` |
| `docs/` | the design text moved out of `docs/ARCHITECTURE.md` (`ARCHITECTURE_monte_carlo.md`) and `docs/CUDA_KERNELS.md` (`CUDA_KERNELS_monte_carlo.md`) |

The extracts are reference copies of code cut out of files that stayed in the build, as those files were at
5bc8f6ce3; each starts with a comment naming its source. They are not compilable units on their own.

The numerics of the retired methods stay in `docs/NUMERICS.md` under retired banners (§0.4, §6.1.2, §6.2–§6.6,
§7, §8.2, §9.1–§9.7, §10.1, §10.3, §10.4, §11.4, §12.3, §12.6.2, §12.7.1–§12.7.2, Appendix A.10), with their input
keys in `docs/SPECIFICATION.md` §6.4.5 and their gates in `docs/VERIFICATION.md` (§5.4, §7.1–§7.3, §8, §9) and
`docs/PERFORMANCE.md` (P1–P3, P5).

## What the build keeps

- Input: `mode="imc_ddmc"` and `enabled=True` of `Radiation.imc`, `.ddmc`, `.holo` or `.imc.difference` are
  ConfigErrors. The keys only the Monte Carlo radiation read are accepted and ignored with a warning:
  `Radiation.imc` except `two_stage`, `Radiation.ddmc`, `Radiation.diffusion`, `Radiation.holo`,
  `Radiation.origin_parity_only`, `Radiation.boundary.marshak_particles`, `Radiation.max_pool_size`,
  `Radiation.momentum_deposition`, `Numerics.dt.f_min_fleck`, `Numerics.safety.opacity_floor` / `opacity_cap`,
  `Diagnostics.mc_stats`, `Diagnostics.fleck_diag`, `Parallel.migration`, and the materials'
  `opacity.lambda_method` / `f_min` (their former defaults pass silently). `Radiation.imc.two_stage` still acts.
- Output (HDF5 schema 2): the history groups `mc/*`, `holo/*`, `difference/*` and the column
  `diagnostics/dt_breakdown_history/dt_rad` are gone, the maximum-principle monitor moved from `mc/overshoot_*` to
  `radiation/overshoot_*` (a history file started before keeps writing it under `mc/`); snapshots and checkpoints
  lost `radiation/ddmc_flag`, `radiation/delta_E_rad_prev`, `holo/`, `difference/`, and checkpoints `particles/` and
  `rng/`. A schema-1 checkpoint is read when its `particles/` is empty (one holding particles was written by the
  retired mode and is refused); the frozen-configuration comparison drops the retired keys on both sides. A restart
  still needs the deck that wrote the checkpoint, unchanged: the frozen configuration holds the deck's hash and
  `Output.directory`.
- Names that outlived the code: `State::holo_ale_invalidated` (the ALE flag that tells the FLD and S_N solvers to
  rebuild their mesh caches), the time-step limiter id 4 and the dt-winner code 2 (unused, the other codes keep their
  values), and bits 3, 6 and 7 of the persistent loop's laser error code (unused).
- `tenryu verify` refuses the retired target names with a message pointing here (exit 1).

## Test status at 5bc8f6ce3

Run on 2026-09-30 on a RunPod RTX 4090 (CUDA 12.6), with 5bc8f6ce3 built in Release (MPI, HDF5 and Python on), each
test executable run from the tree root and each verify target through `tenryu verify <target>`:

- The 34 test executables that moved here all pass.
- Of the 21 verify targets, the twelve that set up their configuration in C++ (the seven `nlte_*`, the four
  `imc_ddmc_*` and `void_passthrough`) pass. The four `parallel_*` targets are stubs that exit 0 without testing.
  `ddmc_diffusion`, `ddmc_leak_normalization`, `ddmc_multigroup` and `mmatrix_fallback` fail before they run: their
  decks do not set `Radiation.mode`, and since the default became `multigroup_diffusion` (2026-04-26, 2d83c33a0) the
  builder reads them as FLD decks and rejects their top-level `Radiation.boundary` faces
  (`Radiation.mode="multigroup_diffusion" ignores the top-level Radiation.boundary face settings`).
  `gxii_1d_regression` has been refused by the verify command since 2026-07-06, and its deck and golden were no
  longer in the tree.

| Kind | Name | Result | Detail |
|---|---|---|---|
| test | `test_boundary_distance_2d` | PASS | 4 cases, 20 assertions |
| test | `test_census_comb` | PASS | 8 cases, 43 assertions |
| test | `test_composite_sort` | PASS | 2 cases, 32 assertions |
| test | `test_ddmc_2d_gpu` | PASS | 19 cases, 131 assertions |
| test | `test_ddmc_coefficients` | PASS | 3 cases, 22 assertions |
| test | `test_ddmc_coefficients_2d` | PASS | 1 case, 17 assertions |
| test | `test_ddmc_gpu` | PASS | 13 cases, 87 assertions |
| test | `test_ddmc_momentum` | PASS | 3 cases, 7 assertions |
| test | `test_ddmc_multigroup` | PASS | 5 cases, 83 assertions |
| test | `test_ddmc_prepare_demotion` | PASS | 1 case, 31 assertions |
| test | `test_deterministic_diffusion_1d` | PASS | 7 cases, 30 assertions |
| test | `test_difference_reference` | PASS | 2 cases, 9 assertions |
| test | `test_diffusion_source_solve` | PASS | 2 cases, 10 assertions |
| test | `test_driver_retry_imc_abort` | PASS | a forked driver with mode `ImcDdmc` and the step retry aborts, as expected |
| test | `test_dt_rad_limit_mode` | PASS | 1 case, 6 assertions |
| test | `test_face_geometry_2d` | PASS | 5 cases, 200125 assertions |
| test | `test_fleck` | PASS | 1 case, 3 assertions |
| test | `test_fleck_nlte` | PASS | 4 cases, 26 assertions |
| test | `test_group_resample_nlte` | PASS | 4 cases, 8 assertions |
| test | `test_holo_lo_solver` | PASS | 12 cases, 75 assertions |
| test | `test_holo_lo_state` | PASS | 1 case, 10 assertions |
| test | `test_holo_selector` | PASS | 9 cases, 62 assertions |
| test | `test_imc_diffusion_diagnostic` | PASS | 1 case, 7 assertions |
| test | `test_imc_source_nlte` | PASS | 5 cases, 16020 assertions |
| test | `test_interface` | PASS | 7 cases, 32 assertions |
| test | `test_mmatrix_check` | PASS | 2 cases, 11 assertions |
| test | `test_mode_selector` | PASS | 8 cases, 35 assertions |
| test | `test_particle_generation_2d` | PASS | 1 case, 250009 assertions |
| test | `test_particle_pool` | PASS | 1 case, 35 assertions |
| test | `test_particle_reid` | PASS | 1 case, 26 assertions |
| test | `test_philox_cpu` | PASS | 1 case, 11 assertions |
| test | `test_rad_lite_mesh_nlte` | PASS | 1 case, 7 assertions |
| test | `test_sn_origin_bc` | PASS | 1 case, 7 assertions |
| test | `test_tally` | PASS | 10 cases, 59 assertions |
| verify | `nlte_sanity` | PASS | rc=0 |
| verify | `nlte_lte_regression` | PASS | rc=0 |
| verify | `nlte_cooling_mms` | PASS | rc=0 |
| verify | `nlte_lambda_agreement` | PASS | rc=0 |
| verify | `nlte_ddmc_classification` | PASS | rc=0 |
| verify | `nlte_energy_conservation` | PASS | rc=0 |
| verify | `nlte_group_resample` | PASS | rc=0 |
| verify | `imc_ddmc_hybrid` | PASS | rc=0 |
| verify | `imc_ddmc_angular` | PASS | rc=0 |
| verify | `imc_ddmc_tau_scan` | PASS | rc=0 |
| verify | `imc_ddmc_convergence` | PASS | rc=0 |
| verify | `ddmc_diffusion` | FAIL | rc=1 — the deck fails validation (see below) |
| verify | `ddmc_leak_normalization` | FAIL | rc=1 — the deck fails validation (see below) |
| verify | `mmatrix_fallback` | FAIL | rc=1 — the deck fails validation (see below) |
| verify | `ddmc_multigroup` | FAIL | rc=1 — the deck fails validation (see below) |
| verify | `void_passthrough` | PASS | rc=0 |
| verify | `gxii_1d_regression` | FAIL | rc=1 — refused by the verify command since 2026-07-06 |
| verify | `parallel_particle_migration` | PASS | rc=0 — a stub that prints `[SKIP] not yet implemented` |
| verify | `parallel_ddmc_leak` | PASS | rc=0 — a stub that prints `[SKIP] not yet implemented` |
| verify | `parallel_interface_boundary` | PASS | rc=0 — a stub that prints `[SKIP] not yet implemented` |
| verify | `parallel_tally_mode` | PASS | rc=0 — a stub that prints `[SKIP] not yet implemented` |

## Restoring

The simplest way to run the Monte Carlo radiation is to build 5bc8f6ce3 (for example in a separate worktree:
`git worktree add ../tenryu-mc 5bc8f6ce3`); its tests and verify targets are registered there. 5bc8f6ce3 is a
commit of the development repository; in the public repository (tenryu-code/TENRYU) the last snapshot that builds
this code is the one of 2026-09-28 ("Update to main@7c445c8dd"), and the snapshots after it carry it here, unbuilt.

Bringing it back into a newer tree means undoing the detachment commit: move the files back (`git mv`), restore the
CMake entries they had at 5bc8f6ce3 (`src/radiation/CMakeLists.txt`, `src/parallel/CMakeLists.txt`,
`src/verification/CMakeLists.txt`, `tests/radiation/CMakeLists.txt`, `tests/coupling/CMakeLists.txt`,
`tests/core/CMakeLists.txt`, `tests/CMakeLists.txt`), put the extracts back into their source files, re-add the
Config members and their builder / freeze handling, the `RadiationMode::ImcDdmc` value, the output datasets and the
schema-1 reader paths, and re-enable the mode in the builder (it has been a ConfigError since 2026-07-10).

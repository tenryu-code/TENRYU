# retired/ale_1d — the retired 1D ALE

This directory keeps the solution-adaptive ALE of 1D_SPH (`Numerics.ale1d`, the "V3" 1D ALE) out of the build.
Nothing here is compiled: there is no CMakeLists.txt, and no file under `src/`, `tests/` or `examples/` refers to these
files.

| | |
|---|---|
| Method | Between hydro steps: feature sensors on the device fields (laser deposition, ablation front, shock, material interfaces, centre); a cell monitor in the normalized mass coordinate (Gaussian feature weights and a spatial monitor), equidistributed into candidate node radii under node constraints (pinned and protected faces, a displacement cap); a min-width-floor candidate (a local span-preserving respace of the narrowest cells); a conservative remap of mass, material masses, electron and ion energies and radiation group energies in the spherical volume coordinate (MUSCL/minmod with a first-order donor fallback); the node velocities by the half-index shift (Benson 1992 §3.5.5); conservation checks with soft and hard tolerances; a two-phase commit (scratch, then the state). The numerics are in `docs/NUMERICS.md` §3.4. |
| Added | 2026-05-01, 0f1071ebe (opt-in, off by default). An earlier 1D ALE had been removed on 2026-04-26 (5e81fe51a): it smeared shocks and preheated the fuel in the GXII 120 J / 6 ns runs. |
| Moved out of the build | 2026-10-02, branch `lane/retire-1d-ale-20261002` |
| Last commit where it built and its tests ran | **6d62bf929** (see "Test status at 6d62bf929") |

Why it was retired: no deck of `examples/`, the verification set or the Studio presets enabled it; it refused
\(S_N\) radiation, burn and more than one material, so it could not run an ICF capsule deck; the record of 2026-08-07
(`docs/NUMERICS.md` §3.4.1, "現状") says accepted candidates did not reach application on the bench decks; and the
cells of a 1D spherical Lagrangian mesh do not tangle, so the mesh needs no rezone to survive. 1D_SPH runs pure
Lagrangian: no 1D path rezones the mesh or remaps the state any more.

## Layout

The tree mirrors the repository layout.

| Path | Content |
|---|---|
| `src/hydro/` | 15 files moved whole (`git log --follow` shows their history): `ale_1d_types.cuh` (the step result and skip reasons), `ale_1d_driver.{cuh,cu}` (`apply_ale_1d`: triggers, eligibility, candidate gates, the commit), `ale_1d_sensor.{cuh,cu}` (features), `ale_1d_rezone.{cuh,cu}` (the host candidate: monitor, equidistribution, floor candidate), `ale_1d_rezone_device.{cuh,cu}` (the same candidate on the device, bit-equal to the host one), `ale_1d_remap.{cuh,cu}` (the remap, `remap_v3` and `remap_v3_device`), `ale_1d_velocity_project.{cuh,cu}`, `ale_1d_diagnostics.{cuh,cu}` |
| `src/core/namelist/ale_1d_config.cpp` | extract: `Config::NumericsConfig::Ale1dConfig`, the `Numerics.ale1d` parsing of the builder, `validate_ale1d_config`, the frozen-configuration serialization and the defaults a frozen configuration written before a key existed took |
| `src/coupling/driver_ale_1d.cpp` | extract: the call in `Driver::run` and the parts of the driver and of `State` that served only the 1D ALE |
| `tests/hydro/` | 7 test executables moved whole: `test_ale_1d_skeleton`, `test_ale_1d_sensor`, `test_ale_1d_rezone`, `test_ale_1d_rezone_device`, `test_ale_1d_remap`, `test_ale_1d_velocity_project`, `test_ale_1d_driver` |
| `tests/core/test_namelist_ale_1d.cpp` | extract: the 1D ALE cases of `test_namelist_ai_review_1d_keys.cpp` (2 test cases) and `test_namelist_radiation_guards.cpp` (1 section) |
| `examples/` | three decks that run it at 6d62bf929: `noh_ale_1d.py`, `noh_ale_1d_floor.py`, `sedov_ale_1d.py` |
| `docs/` | the text moved out of `docs/ARCHITECTURE.md` (`ARCHITECTURE_ale_1d.md`), `docs/CUDA_KERNELS.md` (`CUDA_KERNELS_ale_1d.md`) and `docs/SPECIFICATION.md` (`SPECIFICATION_ale_1d.md`: the input keys and their defaults) |

The extracts are reference copies of code cut out of files that stayed in the build, as those files were at
6d62bf929; each starts with a comment naming its sources and line ranges. They are not compilable units on their own.

`retired/radiation_monte_carlo/tests/hydro/test_ale_1d_skeleton_imc.cu` (the 1D ALE skeleton test of the Monte Carlo
radiation) belongs to the Monte Carlo retirement and stays there.

## What the build keeps

- Input: `Numerics.ale1d` is accepted and has no effect. A warning names its keys other than `enabled=False`;
  `enabled=True` is a ConfigError, and so are a value that is not a dict and an `enabled` that is not a bool.
  Values that the builder validated before even with the 1D ALE disabled (`every_n_steps < 1`, a soft tolerance
  above its hard one, ...) are no longer errors. The `conservative_advection` coupling of the FLD no longer names
  the 1D ALE in its error.
- Output: no HDF5 schema change (the 1D ALE wrote no dataset of its own). The frozen configuration no longer holds
  `numerics.ale1d`; the restart comparison drops it from a checkpoint written before, which then compares equal
  (its run had the 1D ALE disabled: a deck with it enabled is refused when it is read, so a checkpoint written
  with the 1D ALE on cannot be continued).
- Names that outlived the code: `State::ale_rezoned`, `State::ale_rezone_invocations` and
  `State::ale_last_applied_step` (with the checkpoint's `time_state/ale_last_applied_step` and the history's
  `mesh/ale_rezone_invocations`) are written by the 2D_RZ ALE; in 1D they keep their initial values. The entropy
  ledger's ALE stage belongs to the 2D ALE as well. `State::holo_ale_invalidated`, which the 1D ALE set after a
  commit, is still read by the 1D linear-discontinuous \(S_N\) (it reseeds its histories when the flag is set) and
  cleared by the 1D FLD and \(S_N\) solvers; nothing in 1D sets it now. `Hydro1D::close_eos_and_sound_speed`, added
  for the commit of the 1D ALE (2026-09-23), stays: the closure tests of `test_hydro_1d_step` call it.

## Test status at 6d62bf929

Run on 2026-10-02 on a RunPod RTX 4090 (CUDA 12.6), with 6d62bf929 built in Release, each test executable run whole
from its ctest working directory and each deck through `tenryu run`:

| Kind | Name | Result | Detail |
|---|---|---|---|
| test | `test_ale_1d_skeleton` | PASS | 4 cases, 17 assertions |
| test | `test_ale_1d_sensor` | PASS | 7 cases, 20 assertions |
| test | `test_ale_1d_rezone` | PASS | 16 cases, 870 assertions |
| test | `test_ale_1d_rezone_device` | PASS | 2 cases, 260 assertions (54 random rezone and 32 random floor candidates, device against host, bit for bit) |
| test | `test_ale_1d_remap` | PASS | 13 cases, 566 assertions |
| test | `test_ale_1d_velocity_project` | PASS | 13 cases, 556 assertions |
| test | `test_ale_1d_driver` | PASS | 19 cases, 830 assertions |
| test | `test_namelist_ai_review_1d_keys` | PASS | 7 cases, 38 assertions (two of them are the 1D ALE cases now in `tests/core/test_namelist_ale_1d.cpp`) |
| test | `test_namelist_radiation_guards` | PASS | 7 cases, 24 assertions (one section is the 1D ALE section now in `tests/core/test_namelist_ale_1d.cpp`) |
| deck | `examples/noh_ale_1d.py` | rc=0 | 600 steps, 12 rezones applied (one every 50 steps, the minimum spacing `min_steps_between_ale`) |
| deck | `examples/noh_ale_1d_floor.py` | rc=0 | 600 steps, 12 rezones applied (the first within the first two steps, when only the min-width-floor trigger fired) |
| deck | `examples/sedov_ale_1d.py` | rc=0 | 69 steps to `t_end`, 1 rezone applied (the cadence attempt at step 25; the one at step 50 came too soon after it) |

## Restoring

The simplest way to run the 1D ALE is to build 6d62bf929 (for example in a separate worktree:
`git worktree add ../tenryu-ale-1d 6d62bf929`); its tests are registered there, and the decks in `examples/` run it.

Bringing it back into a newer tree means undoing the retirement commit: move the files back (`git mv`), restore the
CMake entries they had at 6d62bf929 (the 15 file names in `tenryu_hydro` of `src/hydro/CMakeLists.txt`; the 7
`add_executable` / `target_link_libraries` blocks and the 7 `catch_discover_tests` of `tests/hydro/CMakeLists.txt`),
paste the extracts back where their headers say they came from, restore `State::ale1d_floor_cooldown_remaining`, the
builder's `Numerics.ale1d` parsing and the frozen-configuration serialization (a checkpoint written between the
retirement and the restoration has no `numerics.ale1d`, which then compares unequal unless legacy defaults are
added for it), and the documentation (`docs/NUMERICS.md` §3.4, `docs/SPECIFICATION.md` §6.4.7 and §9.1,
`docs/ARCHITECTURE.md`, `docs/CUDA_KERNELS.md`).

# The 1D ALE in docs/ARCHITECTURE.md (retired)

This is the module entry of the 1D ALE as it stood in `docs/ARCHITECTURE.md` §4 (hydro) at 6d62bf929, the last
commit where the code was built. The 1D ALE was retired on 2026-10-02 (see `../README.md`); the paths below are the
ones it had in the build, and the files are now under `retired/ale_1d/src/hydro/`.

- `Hydro::ALE1D`（`src/hydro/ale_1d_driver.{cuh,cu}`, `ale_1d_types.cuh`, `ale_1d_sensor.{cuh,cu}`, `ale_1d_rezone.{cuh,cu}`, `ale_1d_rezone_device.{cuh,cu}`, `ale_1d_remap.{cuh,cu}`, `ale_1d_velocity_project.{cuh,cu}`, `ale_1d_diagnostics.{cuh,cu}`）
  - 1D_SPH solution-adaptive ALE V3 の public API と skip-path diagnostics を保持する。現行版では GPU sensor が `compute_features` で feature list を構築し、`ale_1d_rezone_device` が monitor、common node mask、equidistribution の candidate（min-width floor の candidate、境界節点、幾何の検査、音響 dt の上限まで）を device で構築し（host 実装 `ale_1d_rezone` と同じ演算順・丸めでビット一致、exp・pow は `core::glibc_libm`。2026-10-02 まで host。`TENRYU_ALE1D_HOST=1` で host 実装）、candidate は device のまま remap（`remap_v3_device`）と commit へ渡り、`ale_1d_remap` が volume-coordinate MUSCL/minmod remap（first-order donor fallback と cosine protected-face taper 付き）を caller-owned scratch に書き、`ale_1d_velocity_project` が mass remap の受理済み面質量流束（`Ale1dRemapScratch::mass_flux`）と mass phi を使い、各セルの両端節点速度の組を比量として移流する half-index-shift 法（Benson 1992 §3.5.5）で節点速度 scratch を構築する（2026-09-23 に cell momentum remap + mass-weighted node projection から置換）。`ale_1d_diagnostics` が scratch diagnostics を評価し、`apply_ale_1d` は hard 許容誤差を満たした場合だけ二相 commit で state を更新する。
  - V3 1D ALE は opt-in / experimental。既定は `numerics.ale1d.enabled=False` で、通常の GXII short-pulse は pure Lagrangian を使う。
  - runtime scope は 1D_SPH + deterministic radiation (FLD/S_N) のみ（モンテカルロ輻射は 2026-09-29 にビルドから外した）。

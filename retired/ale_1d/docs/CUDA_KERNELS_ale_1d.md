# The 1D ALE in docs/CUDA_KERNELS.md (retired)

The row of the 1D ALE in the table of 1D production kernels of `docs/CUDA_KERNELS.md` §0 as it stood at 6d62bf929,
the last commit where the code was built. The 1D ALE was retired on 2026-10-02 (see `../README.md`); the file is
now `retired/ale_1d/src/hydro/ale_1d_rezone_device.cu`. The other kernels of the 1D ALE (sensor reductions, remap,
velocity projection, diagnostics) are in the other `ale_1d_*.cu` files there (`grep __global__`).

| モジュール | ファイル | 主なカーネル |
|---|---|---|
| 1D ALE の再配置 candidate | `hydro/ale_1d_rezone_device.cu` | `mass_map_kernel`・`node_mask_kernel`・`monitor_*_kernel`・`rezone_gates_kernel`・`equidistribute_kernel`・`candidate_radii_kernel`・`displacement_kernel`・`floor_candidate_kernel`・`finalize_kernel` ほか |

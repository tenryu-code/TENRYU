# tenryu_plot

Standard 1D post-processing plots for TENRYU output directories.

**Full manual: `docs/POSTPROCESSING.md`** (data preparation, all subcommands,
exact-reference overlays and their caveats, recipes, troubleshooting,
extension guide). This README is the quick reference.

Launch forms:

```bash
tools/tenryu-plot --help
PYTHONPATH=tools python3 -m tenryu_plot --help
```

Dependencies are loaded at use time: `numpy`, `h5py`, and `matplotlib`.

## Commands

Profile at the final snapshot:

```bash
tools/tenryu-plot profile outputs/gxii_solid_1D_fld
```

Profile with an exact reference overlay:

```bash
tools/tenryu-plot profile outputs/noh_spherical --overlay noh --overlay-params rho0=1,v0=1,gamma=1.6666666667
```

RMtV verification run overlay:

```bash
tools/tenryu-plot profile build/plot1d_gate_rmtv --overlay rmtv
```

The RMtV overlay is a frozen exact-reference instance for the `rmtv_1d`
verification deck parameters baked into `src/verification/rmtv_reference_table.hpp`.
Use it only for matching RMtV runs. The certified similarity band is
`XiCertifiedLo..XiFront`; the C++ L2 gate uses `xi` in `[0.40, 1.90]`.

History diagnostics:

```bash
tools/tenryu-plot history outputs/gxii_solid_1D_fld --panels energy,laser
```

Spacetime map:

```bash
tools/tenryu-plot spacetime outputs/gxii_solid_1D_fld --field rho --log --mesh 20
```

Summary bundle:

```bash
tools/tenryu-plot summary outputs/gxii_solid_1D_fld -o /tmp/tenryu_plot_summary
```

Compare two runs:

```bash
tools/tenryu-plot compare outputs/gxii_solid_1D_fld outputs/gxii_solid_1D_fld_001
```

Convergence against an exact reference:

```bash
tools/tenryu-plot convergence outputs/noh_128 outputs/noh_256 --ref noh --field rho
```

Multigroup spectrum:

```bash
tools/tenryu-plot spectrum outputs/gxii_solid_1D_fld --at-r 0.01,0.02
```

Laser diagnostics:

```bash
tools/tenryu-plot laser outputs/gxii_solid_1D_fld --times 0.5ns,1.0ns,1.5ns
```

Design reference: `docs/design/plot1d_standard_package_20260709.md`.

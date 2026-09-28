# Figures of the documentation site

These scripts draw the figures in `docs/site/assets/`. Each one computes what it draws from the formulas and data that the corresponding page states (ray equation, Bouguer invariant, the ion-acoustic response, the SNB spectral primitive, the revolved-volume corner vectors, the mesh generators, the example decks), prints numerical checks, and exits with a non-zero status if a check fails.

Run from the repository root with a Python that has numpy, scipy, matplotlib and PIL:

```sh
export MPLCONFIGDIR="$PWD/tmp/site_figures_mplconfig"
python tools/site_figures/<script>.py
```

Text uses the Hiragino Sans font (macOS); on another system install a font with Latin and Japanese glyphs and change `font.family` in the script. Every PNG is written at twice its display size (`figsize = display size / 100` inches, 200 dpi); the `width` and `height` attributes of the `<img>` tags on the pages are the display size.

| Script | Output (`docs/site/assets/`) | Page |
|---|---|---|
| `corona_rays.py` | — (ray tracing in the model corona, shared) | — |
| `laser_chord_geometry.py` | `laser-chord-geometry.png` | physics/laser |
| `cbet_port_section_geometry.py` | `cbet-port-section-geometry.png` | physics/cbet |
| `cbet_pair_exchange_detuning.py` | `cbet-pair-exchange-detuning.png` | physics/cbet |
| `mesh_1d_regions_corona.py` | `mesh-1d-regions-corona.png` | physics/mesh/1d |
| `mesh_2d_family_gallery.py` | `mesh-2d-family-gallery.png` | physics/mesh/2d |
| `hydro_staggered_rz.py` | `hydro-staggered-rz.png` | physics/hydrodynamics |
| `ale_remap_swept_region.py` | `ale-remap-swept-region.png` | physics/mesh/ale-remap |
| `ale_remap_cycle.py` | `ale-remap-cycle.png` | physics/mesh/ale-remap |
| `overview_numerics_step_flow.py` | `overview-numerics-step-flow.png` | overview/numerics-foundations |
| `burn_reaction_partition.py` | `burn-reaction-partition-en.png`, `-ja.png` | physics/burn |
| `eos_opacity_flow.py` | `eos-opacity-flow.png` | physics/eos-opacity |
| `conduction_flux_composition.py` | `conduction-flux-composition.png` | physics/conduction |
| `conduction_snb_group_ladder.py` | `conduction-snb-group-ladder.png` | physics/conduction |
| `radiation_fld_group_fleck_flow.py` | `radiation-fld-group-fleck-flow.png` | physics/radiation/fld |
| `radiation_fld_rz_five_point.py` | `radiation-fld-rz-five-point.png` | physics/radiation/fld |
| `radiation_sn_rz_ordinates.py` | `radiation-sn-rz-ordinates.png` | physics/radiation/sn |
| `mpi_owned_window_decomposition.py` | `mpi-owned-window-decomposition-en.png`, `-ja.png` | physics/mpi |
| `hot_electron_generation_deposition.py` | `hot-electron-generation-deposition-en.png`, `-ja.png` | physics/hot-electrons |

The remaining assets (`laser-beam-optics.png`, `overview-architecture-flow.png`, `hydro-midpoint-v1-flow.png`) have no script here.

When a page changes a formula, a default or a name that a figure shows, change the script and regenerate the figure in the same commit.

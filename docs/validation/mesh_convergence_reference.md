# TENRYU mesh-convergence reference

Empirical grid-convergence reference for the 1D laser-ablation mesh generator (campaign design: docs/design/mesh_convergence_campaign_20260903.md). Each case was run over a ladder of surface areal-mass zonings; `a_conv` is the coarsest surface cell areal mass [g/cm^2] for which every finer level stays within the tolerances (P_a 5 %, m_abl 10 %, E_abs 3 %, t_bo 2 %; rho_R 10 % in the strict variant) of the finest level. `Predicted ceiling` is the formation-band ceiling of the a-priori resolution-requirement model AS CONFIGURED IN THE RUNS (zones_per_scale_length 8; intensity correction none / not recorded); `r_c` = a_conv / predicted ceiling. Calibration keys in the JSON payload are relative to those recorded parameters; the adopted defaults are documented in docs/design/mesh_resolution_requirement_20260903.md §7.3.

| ID | Description | lambda [nm] | Waveform | rho0 profile | verdict obs. | rho_c [g/cc] | Predicted ceiling [g/cm^2] | a_conv nominal [g/cm^2] | a_conv achieved [g/cm^2] | a_conv lenient | a_conv strict (with rho_R) | r_c | r_c strict | last increment P_a | rate | Levels run | Converged? | Notes |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| C01 | planar solid CD foil | 351 | W1 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.0732e-06 | 1e-06 | 9.6623e-07 | 9.6623e-07 | 9.6623e-07 | 0.9003 | 0.9003 | 0.042397 | 0.74116 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C02 | planar solid CD foil | 351 | W2 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | 2e-06 | 1.9256e-06 | 1.9256e-06 | 9.6623e-07 | 1.244 | 0.62423 | 0.020506 | 0.55815 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C03 | planar solid CD foil | 351 | W3 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 2.3122e-06 | 1e-06 | 9.6623e-07 | 9.6623e-07 | 9.6623e-07 | 0.41788 | 0.41788 | 0.015119 | 0.50116 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C04 | planar solid CD foil | 527 | W1 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.013331 | 6.2425e-07 | 1e-06 | 9.6623e-07 | 9.6623e-07 | 9.6623e-07 | 1.5478 | 1.5478 | 0.027784 | 0.42866 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C05 | planar solid CD foil | 527 | W2 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.013331 | 9.0032e-07 | 1e-06 | 9.6623e-07 | 9.6623e-07 | — | 1.0732 | — | 0.023211 | 0.45546 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C06 | planar solid CD foil | 527 | W3 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.013331 | 1.3449e-06 | — | — | 4.8684e-07 | — | — | — | 0.035409 | 0.4434 | 0, 1, 2, 3, 4, 5, 6 | no | reference not converged |
| C07 | planar solid CD foil | 1053 | W1 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.0033392 | 2.4805e-07 | — | — | 1.2334e-07 | — | — | — | 0.041493 | 0.39229 | 0, 1, 2, 3, 4, 5, 6, 7, 8 | no | missing: t_bo; reference not converged |
| C08 | planar solid CD foil | 1053 | W2 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.0033392 | 3.5775e-07 | — | — | 1.2334e-07 | — | — | — | 0.051437 | 0.48137 | 0, 1, 2, 3, 4, 5, 6, 7, 8 | no | reference not converged |
| C09 | planar solid CD foil | 1053 | W3 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.0033392 | 5.344e-07 | — | — | 4.8684e-07 | — | — | — | 0.11205 | 0.17407 | 0, 1, 2, 3, 4, 5, 6 | no | reference not converged |
| C10 | planar solid CD foil, 3e14 W/cm^2 Gaussian, 1.0 ns FWHM at 1.2 ns | 351 | W4 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 2.1791e-06 | 2e-06 | 1.7979e-06 | 1.7979e-06 | 9.0251e-07 | 0.82503 | 0.41416 | 0.011696 | 0.49348 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C11 | planar solid CD foil, 2e13 W/cm^2 foot then 3e14 W/cm^2 main | 351 | W5 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 2.0795e-06 | 2e-06 | 1.7979e-06 | 1.7979e-06 | — | 0.86455 | — | 0.02104 | 0.50237 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C12 | planar solid CD foil, 100 ps Gaussian picket then 3e14 W/cm^2 main | 351 | W6 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.5656e-06 | 1e-06 | 9.6623e-07 | 9.6623e-07 | 9.6623e-07 | 0.61717 | 0.61717 | 0.031926 | 0.91079 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C13 | planar solid CD foil, linear ramp to 5e14 W/cm^2 over 2.0 ns | 351 | W7 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 2.3427e-06 | 2e-06 | 1.7979e-06 | 1.7979e-06 | 9.0251e-07 | 0.76743 | 0.38524 | 0.014659 | 0.52858 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C14 | planar solid CD foil, 3e13 W/cm^2, 100 ps rise, long flat | 351 | W8 | solid:1.05 g/cc x 24.7619 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.9414e-06 | 2e-06 | 1.9256e-06 | 1.9256e-06 | 1.9256e-06 | 0.99186 | 0.99186 | 0.022987 | 0.57137 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C15 | planar CD density variant rho=0.05 g/cc | 351 | W2 | CD:0.05 g/cc x 520 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | — | — | 7.6513e-06 | — | — | — | 0.096214 | 4.055 | 0, 1, 2, 3, 4, 5, 6 | no | missing: t_bo; reference not converged |
| C16 | planar CD density variant rho=0.2 g/cc | 351 | W2 | CD:0.2 g/cc x 130 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | 4e-06 | 3.8614e-06 | 3.8614e-06 | 3.8614e-06 | 2.4946 | 2.4946 | 0.01056 | 0.67502 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C17 | planar CD density variant rho=0.5 g/cc | 351 | W2 | CD:0.5 g/cc x 52 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | 4e-06 | 3.8614e-06 | 3.8614e-06 | 9.0251e-07 | 2.4946 | 0.58307 | 0.0070618 | 0.40951 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C18 | planar CD density variant rho=2.5 g/cc | 351 | W2 | CD:2.5 g/cc x 10.4 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | 1e-06 | 9.6623e-07 | 9.6623e-07 | 9.6623e-07 | 0.62423 | 0.62423 | 0.021217 | 0.40197 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C19 | planar CD density variant rho=0.2 g/cc | 1053 | W2 | CD:0.2 g/cc x 130 um | P_a,m_abl,E_abs,t_bo | 0.0033392 | 3.5775e-07 | — | — | 4.8684e-07 | — | — | — | 0.11106 | 0.42107 | 0, 1, 2, 3, 4, 5, 6 | no | missing: t_bo; reference not converged |
| C20 | planar CD density variant rho=2.5 g/cc | 1053 | W2 | CD:2.5 g/cc x 10.4 um | P_a,m_abl,E_abs,t_bo | 0.0033392 | 3.5775e-07 | — | — | 4.8684e-07 | — | — | — | 0.25877 | 0.56409 | 0, 1, 2, 3, 4, 5, 6 | no | reference not converged |
| C21 | planar solid CD, 10 um | 351 | W2 | solid:1.05 g/cc x 10 um | m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | 2e-06 | 1.8958e-06 | 1.8958e-06 | 1.8958e-06 | 1.2248 | 1.2248 | 0.059385 | 0.3226 | 0, 1, 2, 3, 4, 5, 6 | yes | P_a excluded: unablated outer-half mask empties within the window |
| C22 | planar solid CD, 50 um | 351 | W2 | solid:1.05 g/cc x 50 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | 4e-06 | 3.87e-06 | 3.87e-06 | 3.87e-06 | 2.5002 | 2.5002 | 0.018914 | 0.78158 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C23 | planar foam-on-solid CD | 351 | W2 | solid:1.05 g/cc x 20 um; foam:0.1 g/cc x 25 um | m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | 2e-06 | 1.9841e-06 | 1.9841e-06 | 1.9841e-06 | 1.2818 | 1.2818 | 0.1154 | 2.0727 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C24 | planar solid-on-foam CD | 351 | W2 | foam:0.1 g/cc x 50 um; solid:1.05 g/cc x 10 um | m_abl,E_abs,t_bo | 0.030053 | 1.5479e-06 | 4e-06 | 3.7266e-06 | 3.7266e-06 | 3.7266e-06 | 2.4076 | 2.4076 | 0.0089756 | 0.49449 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C25 | planar CD density variant rho=0.2 g/cc, foot drive | 351 | W5 | CD:0.2 g/cc x 130 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 2.0795e-06 | 4e-06 | 3.8614e-06 | 3.8614e-06 | 3.8614e-06 | 1.8568 | 1.8568 | 0.0066404 | 0.47143 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C26 | planar CD density variant rho=2.5 g/cc, foot drive | 351 | W5 | CD:2.5 g/cc x 10.4 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 2.0795e-06 | 4e-06 | 3.8614e-06 | 3.8614e-06 | 3.8614e-06 | 1.8568 | 1.8568 | 0.036087 | 1.1147 | 0, 1, 2, 3, 4, 5 | yes |  |
| C27 | planar CD rho=0.2 g/cc, picket drive | 351 | W6 | CD:0.2 g/cc x 130 um | P_a,m_abl,E_abs,t_bo | 0.030053 | 1.5656e-06 | 4e-06 | 3.8614e-06 | 3.8614e-06 | 3.8614e-06 | 2.4664 | 2.4664 | 0.0081154 | 0.36993 | 0, 1, 2, 3, 4, 5, 6 | yes |  |
| C28 | spherical GXII CD shell, Gaussian drive | 527 | W4 | fill:0.02 g/cc x 243 um; shell:1.05 g/cc x 7 um | m_abl,E_abs | 0.013331 | 1.2675e-06 | 8e-06 | 6.6941e-06 | 6.6941e-06 | — | 5.2813 | — | 0.60702 | 0.76801 | 0, 1, 2, 3, 4, 5, 6 | yes | missing: t_bo |
| C29 | spherical GXII CD shell, high-intensity drive | 351 | W3 | fill:0.02 g/cc x 243 um; shell:1.05 g/cc x 7 um | m_abl,E_abs | 0.030053 | 2.3122e-06 | 1e-06 | 8.9389e-07 | 8.9389e-07 | — | 0.38659 | — | 0.77664 | 0.85246 | 0, 1, 2, 3, 4, 5, 6 | yes | P_a excluded: unablated outer-half mask empties within the window; missing: t_bo |

## Sanity studies

| ID | Level | Varied quantity | Value | P_a deviation | m_abl deviation | E_abs deviation | rho_R deviation | t_bo deviation |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| C02-S1 | 0 | interior_areal_mass_g_cm2 | 2e-05 | 0.0091586 | 0.0059309 | 0.00063721 | 0.11601 | 0.020408 |
| C02-S1 | 1 | interior_areal_mass_g_cm2 | 1e-05 | 0.0089208 | 0.0027643 | 0.0015735 | 0.059036 | 0 |
| C02-S1 | 2 | interior_areal_mass_g_cm2 | 5e-06 | 0 | 0 | 0 | 0 | 0 |
| C02-S2 | 0 | cfl_hydro | 0.3 | 0.012968 | 0.0080299 | 0.011405 | 0.040239 | 0 |
| C02-S2 | 1 | cfl_hydro | 0.15 | 0 | 0 | 0 | 0 | 0 |

## Calibration

- Converged cases: 22
- Geometric mean r_c: 1.2616
- Geometric mean r_c lenient: 1.166
- Geometric mean r_c strict (with rho_R): 1.0543
- r_c spread (min/max): 0.38659 / 5.2813
- Recorded zones_per_scale_length: 8
- Mean-matched zones_per_scale_length: 6.3412
- All-cases-safe zones_per_scale_length: 20.694

### By wavelength_nm

- 351: 1.1674
- 527: 2.0625

### By waveform

- W1: 1.1805
- W2: 1.541
- W3: 0.40193
- W4: 2.0874
- W5: 1.4392
- W6: 1.2338
- W7: 0.76743
- W8: 0.99186

### By density

- 0.02/1.05: 1.4289
- 0.1/1.05: 2.4076
- 0.2: 2.2522
- 0.5: 2.4946
- 1.05: 0.98135
- 1.05/0.1: 1.2818
- 2.5: 1.0766

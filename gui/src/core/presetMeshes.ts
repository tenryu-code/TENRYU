// Recommended meshes of the laser-driven 1D presets (presets1d.ts), made by tools/assist
// recommend-mesh --deck <preset deck> --tenryu <binary> and written by
// scripts/make_preset_meshes.sh. Regenerate after changing a preset's target, drive or physics
// switches: the conditions key below must equal recommendationConditionsKey(preset) (a unit test
// checks it), otherwise Studio reports the preset's mesh as stale.

export type PresetMeshId =
  | "laserFoil"
  | "laserShell"
  | "snbFoil"
  | "cbetHotElectrons"
  | "tableRadiationFoil"
  | "coldEquilibriumLayers";

export interface PresetMesh {
  conditionsKey: string;
  createdAt: string;
  /** The solver binary the recommendation was validated with. */
  binary: string;
  /** The parts of the recommend-mesh JSON that parseRecommendation reads. */
  payload: unknown;
}

export const PRESET_MESHES: Partial<Record<PresetMeshId, PresetMesh>> = {
  cbetHotElectrons: {
    conditionsKey: "{\"geometry\":\"spherical\",\"tEnd\":1.2e-9,\"temperatureModel\":\"2T\",\"rMin\":0,\"rMax\":0.04,\"regions\":[[\"D2\",0.023,0.0025],[\"CD\",0.025,1.05]],\"vacuum\":true,\"corona\":null,\"materials\":[[\"D2\",2.01,1,\"ideal_gas\",\"\"],[\"CD\",7,3.5,\"ideal_gas\",\"\"]],\"zbar\":null,\"radiation\":false,\"laser\":{\"wavelength\":351,\"waveform\":{\"square\":[3926990816987.2417,1e-9,1.0000000000000002e-10,1.0000000000000002e-10]}},\"conduction\":[true,\"implicit\"]}",
    createdAt: "2026-10-01",
    binary: "tenryu main@efa376c7b (CUDA 12.6 build)",
    payload: {
      "mesh": {
        "geometry_1d": "spherical",
        "r_max": 0.04,
        "r_min": 0,
        "resolution_requirement": {
          "apply": "enforce"
        },
        "zoning_intent": {
          "bands": [
            {
              "cell_measure_max": 1.2630459104492405e-7,
              "measure_frac_begin": 0.008307801658312906,
              "measure_frac_end": 0.07575696274642024
            },
            {
              "cell_measure_max": 6.287750043100754e-9,
              "measure_frac_begin": 0.07575696274642024,
              "measure_frac_end": 0.9999999867875431
            }
          ],
          "density_regions": [
            {
              "r_end": 0.023,
              "rho": 0.0025
            },
            {
              "r_end": 0.025,
              "rho": 1.05
            },
            {
              "r_end": 0.04,
              "rho": 1e-9
            }
          ],
          "dr_min": 3.6592592592592625e-8,
          "measure": "spherical_cell_mass",
          "min_cells_per_segment": 40,
          "n_cells": 2600,
          "pins": [
            {
              "r": 0.023,
              "ratio_jump_allowed": true
            },
            {
              "r": 0.025,
              "ratio_jump_allowed": true
            }
          ],
          "preferred_ratio": 1.3,
          "profile": [
            {
              "r": 0,
              "w": 9.745878590041286e-9
            },
            {
              "r": 0.007749143323805882,
              "w": 9.745878590041286e-9
            },
            {
              "r": 0.007749151072956955,
              "w": 2.923763577012386e-8
            },
            {
              "r": 0.009298971988567058,
              "w": 2.923763577012386e-8
            },
            {
              "r": 0.009298981287548346,
              "w": 4.210219550897836e-8
            },
            {
              "r": 0.01115876638628047,
              "w": 4.210219550897836e-8
            },
            {
              "r": 0.011158777545058014,
              "w": 6.062716153292884e-8
            },
            {
              "r": 0.013390519663536563,
              "w": 6.062716153292884e-8
            },
            {
              "r": 0.013390533054069616,
              "w": 8.73031126074175e-8
            },
            {
              "r": 0.016068623596243874,
              "w": 8.73031126074175e-8
            },
            {
              "r": 0.01606863966488354,
              "w": 1.257164821546812e-7
            },
            {
              "r": 0.019282348315492647,
              "w": 1.257164821546812e-7
            },
            {
              "r": 0.019282367597860246,
              "w": 1.810317343027409e-7
            },
            {
              "r": 0.022999977,
              "w": 1.810317343027409e-7
            },
            {
              "r": 0.023,
              "w": 1.20934052010674e-7
            },
            {
              "r": 0.023138841117432293,
              "w": 1.20934052010674e-7
            },
            {
              "r": 0.023147231892318657,
              "w": 1.20934052010674e-7
            },
            {
              "r": 0.023147255039573697,
              "w": 6.020391534873059e-9
            },
            {
              "r": 0.02344417967191048,
              "w": 6.020391534873059e-9
            },
            {
              "r": 0.023733767788184642,
              "w": 6.020391534873059e-9
            },
            {
              "r": 0.02401645643748351,
              "w": 6.020391534873059e-9
            },
            {
              "r": 0.024292642443902723,
              "w": 6.020391534873059e-9
            },
            {
              "r": 0.02456268732786667,
              "w": 6.020391534873059e-9
            },
            {
              "r": 0.02482692148579994,
              "w": 6.020391534873059e-9
            },
            {
              "r": 0.025,
              "w": 6.020391534873059e-9
            }
          ],
          "ratio_hard_max": 1.3
        }
      },
      "recommendation": {
        "surface_areal_mass_g_cm2": 8.427170528722215e-7,
        "mode": "apriori_fallback"
      },
      "evidence": [
        {
          "id": "C29"
        },
        {
          "id": "C28"
        },
        {
          "id": "C24"
        },
        {
          "id": "C02"
        },
        {
          "id": "C12"
        }
      ],
      "flags": [
        "extrapolation"
      ],
      "warnings": [
        "material outside CD",
        "material A/Z outside campaign CD average atom (A=7, Z=3.5)",
        "shell radius, thickness or fill outside GXII cases",
        "physics differs from radiation-off CD tmat, 2T implicit conduction"
      ],
      "confidence": "outside campaign; convergence pair required",
      "validation": {
        "status": "validated",
        "achieved_surface_areal_mass_g_cm2": 7.579122038755011e-7
      }
    },
  },
  coldEquilibriumLayers: {
    conditionsKey: "{\"geometry\":\"planar\",\"tEnd\":1.5000000000000002e-9,\"temperatureModel\":\"2T\",\"rMin\":0,\"rMax\":0.0175,\"regions\":[[\"D2\",0.005,0.17],[\"CD\",0.0075,1.05]],\"vacuum\":true,\"corona\":null,\"materials\":[[\"D2\",2.01,1,\"tmat\",\"TMAT-H5/D2.tmat.h5\"],[\"CD\",7,3.5,\"tmat\",\"TMAT-H5/CD.tmat.h5\"]],\"zbar\":null,\"radiation\":false,\"laser\":{\"wavelength\":351,\"waveform\":{\"square\":[100000000000000,1e-9,1.0000000000000002e-10,1.0000000000000002e-10]}},\"conduction\":[true,\"implicit\"]}",
    createdAt: "2026-10-01",
    binary: "tenryu main@efa376c7b (CUDA 12.6 build)",
    payload: {
      "mesh": {
        "geometry_1d": "planar",
        "r_max": 0.0175,
        "r_min": 0,
        "resolution_requirement": {
          "apply": "enforce"
        },
        "zoning_intent": {
          "bands": [
            {
              "cell_measure_max": 0.000019,
              "measure_frac_begin": 0,
              "measure_frac_end": 0.6962771273151093
            },
            {
              "cell_measure_max": 8.912578905212104e-7,
              "measure_frac_begin": 0.6962771273151093,
              "measure_frac_end": 0.9999999971223021
            }
          ],
          "density_regions": [
            {
              "r_end": 0.005,
              "rho": 0.17
            },
            {
              "r_end": 0.0075,
              "rho": 1.05
            },
            {
              "r_end": 0.0175,
              "rho": 1e-9
            }
          ],
          "dr_min": 4.244085192958145e-8,
          "measure": "areal_mass",
          "min_cells_per_segment": 40,
          "n_cells": 1509,
          "pins": [
            {
              "r": 0.005,
              "ratio_jump_allowed": true
            },
            {
              "r": 0.0075,
              "ratio_jump_allowed": true
            }
          ],
          "preferred_ratio": 1.3,
          "profile": [
            {
              "r": 0,
              "w": 0.000009770114942528732
            },
            {
              "r": 0.004999995,
              "w": 0.000009770114942528732
            },
            {
              "r": 0.005,
              "w": 0.000018179457320803056
            },
            {
              "r": 0.0064948154331616655,
              "w": 0.000018179457320803056
            },
            {
              "r": 0.006494821927983593,
              "w": 8.527676201347003e-7
            },
            {
              "r": 0.0066455986387860535,
              "w": 8.527676201347003e-7
            },
            {
              "r": 0.006796375349588515,
              "w": 8.527676201347003e-7
            },
            {
              "r": 0.006947152060390976,
              "w": 8.527676201347003e-7
            },
            {
              "r": 0.007097928771193437,
              "w": 8.527676201347003e-7
            },
            {
              "r": 0.007248705481995899,
              "w": 8.527676201347003e-7
            },
            {
              "r": 0.0073994821927983595,
              "w": 8.527676201347003e-7
            },
            {
              "r": 0.0075,
              "w": 8.527676201347003e-7
            }
          ],
          "ratio_hard_max": 1.3
        }
      },
      "recommendation": {
        "surface_areal_mass_g_cm2": 9.381662005486425e-7,
        "mode": "apriori_fallback"
      },
      "evidence": [
        {
          "id": "C24"
        },
        {
          "id": "C01"
        },
        {
          "id": "C22"
        },
        {
          "id": "C04"
        },
        {
          "id": "C17"
        }
      ],
      "flags": [
        "extrapolation"
      ],
      "warnings": [
        "material outside CD",
        "material A/Z outside campaign CD average atom (A=7, Z=3.5)",
        "physics differs from radiation-off CD tmat, 2T implicit conduction"
      ],
      "confidence": "outside campaign; convergence pair required",
      "validation": {
        "status": "validated",
        "achieved_surface_areal_mass_g_cm2": 8.502389084624404e-7
      }
    },
  },
  laserFoil: {
    conditionsKey: "{\"geometry\":\"planar\",\"tEnd\":1.5000000000000002e-9,\"temperatureModel\":\"2T\",\"rMin\":0,\"rMax\":0.0125,\"regions\":[[\"CH\",0.0025,1.05]],\"vacuum\":true,\"corona\":null,\"materials\":[[\"CH\",6.5,3.5,\"ideal_gas\",\"\"]],\"zbar\":null,\"radiation\":false,\"laser\":{\"wavelength\":351,\"waveform\":{\"square\":[100000000000000,1e-9,1.0000000000000002e-10,1.0000000000000002e-10]}},\"conduction\":[true,\"implicit\"]}",
    createdAt: "2026-10-01",
    binary: "tenryu main@efa376c7b (CUDA 12.6 build)",
    payload: {
      "mesh": {
        "geometry_1d": "planar",
        "r_max": 0.0125,
        "r_min": 0,
        "resolution_requirement": {
          "apply": "enforce"
        },
        "zoning_intent": {
          "bands": [
            {
              "cell_measure_max": 0.000019,
              "measure_frac_begin": 0,
              "measure_frac_end": 0.6173105019270342
            },
            {
              "cell_measure_max": 8.482950488488966e-7,
              "measure_frac_begin": 0.6173105019270342,
              "measure_frac_end": 0.9999999961904762
            }
          ],
          "density_regions": [
            {
              "r_end": 0.0025,
              "rho": 1.05
            },
            {
              "r_end": 0.0125,
              "rho": 1e-9
            }
          ],
          "dr_min": 4.039500232613793e-8,
          "measure": "areal_mass",
          "min_cells_per_segment": 40,
          "n_cells": 1460,
          "pins": [
            {
              "r": 0.0025,
              "ratio_jump_allowed": true
            }
          ],
          "preferred_ratio": 1.3,
          "profile": [
            {
              "r": 0,
              "w": 0.000018176676417621998
            },
            {
              "r": 0.0015432747174204688,
              "w": 0.000018176676417621998
            },
            {
              "r": 0.0015432762606967295,
              "w": 8.115360320840652e-7
            },
            {
              "r": 0.00168678482159222,
              "w": 8.115360320840652e-7
            },
            {
              "r": 0.0018302933824877108,
              "w": 8.115360320840652e-7
            },
            {
              "r": 0.0019738019433832017,
              "w": 8.115360320840652e-7
            },
            {
              "r": 0.0021173105042786924,
              "w": 8.115360320840652e-7
            },
            {
              "r": 0.0022608190651741836,
              "w": 8.115360320840652e-7
            },
            {
              "r": 0.0024043276260696735,
              "w": 8.115360320840652e-7
            },
            {
              "r": 0.0025,
              "w": 8.115360320840652e-7
            }
          ],
          "ratio_hard_max": 1.3
        }
      },
      "recommendation": {
        "surface_areal_mass_g_cm2": 8.92942156683049e-7,
        "mode": "apriori_fallback"
      },
      "evidence": [
        {
          "id": "C01"
        },
        {
          "id": "C04"
        },
        {
          "id": "C02"
        },
        {
          "id": "C05"
        },
        {
          "id": "C22"
        }
      ],
      "flags": [
        "extrapolation"
      ],
      "warnings": [
        "material outside CD",
        "material A/Z outside campaign CD average atom (A=7, Z=3.5)",
        "physics differs from radiation-off CD tmat, 2T implicit conduction"
      ],
      "confidence": "outside campaign; convergence pair required",
      "validation": {
        "status": "validated",
        "achieved_surface_areal_mass_g_cm2": 7.992550520479099e-7
      }
    },
  },
  laserShell: {
    conditionsKey: "{\"geometry\":\"spherical\",\"tEnd\":2.4e-9,\"temperatureModel\":\"2T\",\"rMin\":0,\"rMax\":0.04,\"regions\":[[\"D2\",0.0243,0.02],[\"CD\",0.025,1.05]],\"vacuum\":true,\"corona\":null,\"materials\":[[\"D2\",2.01,1,\"ideal_gas\",\"\"],[\"CD\",7,3.5,\"ideal_gas\",\"\"]],\"zbar\":null,\"radiation\":false,\"laser\":{\"wavelength\":527,\"waveform\":{\"gaussian\":[\"peak\",2356194490192.345,1e-9,1.2e-9]}},\"conduction\":[true,\"implicit\"]}",
    createdAt: "2026-10-01",
    binary: "tenryu main@efa376c7b (CUDA 12.6 build)",
    payload: {
      "mesh": {
        "geometry_1d": "spherical",
        "r_max": 0.04,
        "r_min": 0,
        "resolution_requirement": {
          "apply": "enforce"
        },
        "zoning_intent": {
          "bands": [
            {
              "cell_measure_max": 5.4513454260915e-9,
              "measure_frac_begin": 0,
              "measure_frac_end": 0.999999970265094
            }
          ],
          "density_regions": [
            {
              "r_end": 0.0243,
              "rho": 0.02
            },
            {
              "r_end": 0.025,
              "rho": 1.05
            },
            {
              "r_end": 0.04,
              "rho": 1e-9
            }
          ],
          "dr_min": 3.0248130370370486e-8,
          "measure": "spherical_cell_mass",
          "min_cells_per_segment": 40,
          "n_cells": 1426,
          "pins": [
            {
              "r": 0.0243,
              "ratio_jump_allowed": true
            },
            {
              "r": 0.025,
              "ratio_jump_allowed": true
            }
          ],
          "preferred_ratio": 1.3,
          "profile": [
            {
              "r": 0,
              "w": 4.570689056349121e-9
            },
            {
              "r": 0.0242999757,
              "w": 4.570689056349121e-9
            },
            {
              "r": 0.0243,
              "w": 5.23070843591457e-9
            },
            {
              "r": 0.02431316773871795,
              "w": 5.23070843591457e-9
            },
            {
              "r": 0.024504320274654625,
              "w": 5.23070843591457e-9
            },
            {
              "r": 0.024692536177166952,
              "w": 5.23070843591457e-9
            },
            {
              "r": 0.024877925699027904,
              "w": 5.23070843591457e-9
            },
            {
              "r": 0.025,
              "w": 5.23070843591457e-9
            }
          ],
          "ratio_hard_max": 1.3
        }
      },
      "recommendation": {
        "surface_areal_mass_g_cm2": 7.30617744053774e-7,
        "mode": "apriori_fallback"
      },
      "evidence": [
        {
          "id": "C28"
        },
        {
          "id": "C29"
        },
        {
          "id": "C24"
        },
        {
          "id": "C26"
        },
        {
          "id": "C10"
        }
      ],
      "flags": [
        "extrapolation"
      ],
      "warnings": [
        "material outside CD",
        "material A/Z outside campaign CD average atom (A=7, Z=3.5)",
        "physics differs from radiation-off CD tmat, 2T implicit conduction"
      ],
      "confidence": "outside campaign; convergence pair required",
      "validation": {
        "status": "validated",
        "achieved_surface_areal_mass_g_cm2": 6.352107377812304e-7
      }
    },
  },
  snbFoil: {
    conditionsKey: "{\"geometry\":\"planar\",\"tEnd\":1.2e-9,\"temperatureModel\":\"2T\",\"rMin\":0,\"rMax\":0.0125,\"regions\":[[\"CH\",0.0025,1.05]],\"vacuum\":true,\"corona\":null,\"materials\":[[\"CH\",6.5,3.5,\"ideal_gas\",\"\"]],\"zbar\":null,\"radiation\":false,\"laser\":{\"wavelength\":351,\"waveform\":{\"square\":[500000000000000,1e-9,1.0000000000000002e-10,1.0000000000000002e-10]}},\"conduction\":[true,\"sts\"]}",
    createdAt: "2026-10-01",
    binary: "tenryu main@efa376c7b (CUDA 12.6 build)",
    payload: {
      "mesh": {
        "geometry_1d": "planar",
        "r_max": 0.0125,
        "r_min": 0,
        "resolution_requirement": {
          "apply": "enforce"
        },
        "zoning_intent": {
          "bands": [
            {
              "cell_measure_max": 0.000019,
              "measure_frac_begin": 0,
              "measure_frac_end": 0.3456101659541324
            },
            {
              "cell_measure_max": 7.619894034915861e-7,
              "measure_frac_begin": 0.3456101659541324,
              "measure_frac_end": 0.9999999961904762
            }
          ],
          "density_regions": [
            {
              "r_end": 0.0025,
              "rho": 1.05
            },
            {
              "r_end": 0.0125,
              "rho": 1e-9
            }
          ],
          "dr_min": 3.628520969007553e-8,
          "measure": "areal_mass",
          "min_cells_per_segment": 40,
          "n_cells": 2600,
          "pins": [
            {
              "r": 0.0025,
              "ratio_jump_allowed": true
            }
          ],
          "preferred_ratio": 1.3,
          "profile": [
            {
              "r": 0,
              "w": 0.00001851013379363232
            },
            {
              "r": 0.0008640245541514313,
              "w": 0.00001851013379363232
            },
            {
              "r": 0.0008640254181768494,
              "w": 7.423434635768073e-7
            },
            {
              "r": 0.0011094216054503222,
              "w": 7.423434635768073e-7
            },
            {
              "r": 0.001354817792723795,
              "w": 7.423434635768073e-7
            },
            {
              "r": 0.0016002139799972677,
              "w": 7.423434635768073e-7
            },
            {
              "r": 0.0018456101672707404,
              "w": 7.423434635768073e-7
            },
            {
              "r": 0.0020910063545442133,
              "w": 7.423434635768073e-7
            },
            {
              "r": 0.0023364025418176858,
              "w": 7.423434635768073e-7
            },
            {
              "r": 0.0025,
              "w": 7.423434635768073e-7
            }
          ],
          "ratio_hard_max": 1.3
        }
      },
      "recommendation": {
        "surface_areal_mass_g_cm2": 8.020941089385118e-7,
        "mode": "apriori_fallback"
      },
      "evidence": [
        {
          "id": "C02"
        },
        {
          "id": "C12"
        },
        {
          "id": "C05"
        },
        {
          "id": "C03"
        },
        {
          "id": "C22"
        }
      ],
      "flags": [
        "extrapolation"
      ],
      "warnings": [
        "material outside CD",
        "material A/Z outside campaign CD average atom (A=7, Z=3.5)",
        "physics differs from radiation-off CD tmat, 2T implicit conduction"
      ],
      "confidence": "outside campaign; convergence pair required",
      "validation": {
        "status": "validated",
        "achieved_surface_areal_mass_g_cm2": 7.22895859204013e-7
      }
    },
  },
  tableRadiationFoil: {
    conditionsKey: "{\"geometry\":\"planar\",\"tEnd\":1.5000000000000002e-9,\"temperatureModel\":\"2T\",\"rMin\":0,\"rMax\":0.0125,\"regions\":[[\"CD\",0.0025,1.05]],\"vacuum\":true,\"corona\":null,\"materials\":[[\"CD\",7,3.5,\"tmat\",\"TMAT-H5/CD.tmat.h5\"]],\"zbar\":null,\"radiation\":true,\"laser\":{\"wavelength\":351,\"waveform\":{\"square\":[100000000000000,1e-9,1.0000000000000002e-10,1.0000000000000002e-10]}},\"conduction\":[true,\"implicit\"]}",
    createdAt: "2026-10-01",
    binary: "tenryu main@efa376c7b (CUDA 12.6 build)",
    payload: {
      "mesh": {
        "geometry_1d": "planar",
        "r_max": 0.0125,
        "r_min": 0,
        "resolution_requirement": {
          "apply": "enforce"
        },
        "zoning_intent": {
          "bands": [
            {
              "cell_measure_max": 0.000019,
              "measure_frac_begin": 0,
              "measure_frac_end": 0.5979287689156145
            },
            {
              "cell_measure_max": 8.912578905212104e-7,
              "measure_frac_begin": 0.5979287689156145,
              "measure_frac_end": 0.9999999961904762
            }
          ],
          "density_regions": [
            {
              "r_end": 0.0025,
              "rho": 1.05
            },
            {
              "r_end": 0.0125,
              "rho": 1e-9
            }
          ],
          "dr_min": 4.244085192958145e-8,
          "measure": "areal_mass",
          "min_cells_per_segment": 40,
          "n_cells": 1457,
          "pins": [
            {
              "r": 0.0025,
              "ratio_jump_allowed": true
            }
          ],
          "preferred_ratio": 1.3,
          "profile": [
            {
              "r": 0,
              "w": 0.000018179434953251516
            },
            {
              "r": 0.0014948204331616643,
              "w": 0.000018179434953251516
            },
            {
              "r": 0.0014948219279835922,
              "w": 8.527665709106582e-7
            },
            {
              "r": 0.0016455986387860536,
              "w": 8.527665709106582e-7
            },
            {
              "r": 0.001796375349588515,
              "w": 8.527665709106582e-7
            },
            {
              "r": 0.0019471520603909762,
              "w": 8.527665709106582e-7
            },
            {
              "r": 0.002097928771193438,
              "w": 8.527665709106582e-7
            },
            {
              "r": 0.002248705481995899,
              "w": 8.527665709106582e-7
            },
            {
              "r": 0.00239948219279836,
              "w": 8.527665709106582e-7
            },
            {
              "r": 0.0025,
              "w": 8.527665709106582e-7
            }
          ],
          "ratio_hard_max": 1.3
        }
      },
      "recommendation": {
        "surface_areal_mass_g_cm2": 9.381662005486425e-7,
        "mode": "apriori_fallback"
      },
      "evidence": [
        {
          "id": "C01"
        },
        {
          "id": "C04"
        },
        {
          "id": "C02"
        },
        {
          "id": "C05"
        },
        {
          "id": "C22"
        }
      ],
      "flags": [
        "extrapolation"
      ],
      "warnings": [
        "physics differs from radiation-off CD tmat, 2T implicit conduction"
      ],
      "confidence": "outside campaign; convergence pair required",
      "validation": {
        "status": "validated",
        "achieved_surface_areal_mass_g_cm2": 8.387236863389471e-7
      }
    },
  },
};

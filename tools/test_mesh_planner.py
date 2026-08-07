#!/usr/bin/env python3
"""Unit tests for the pure-Python TENRYU mesh composition planner."""

import math
import unittest

from mesh_planner import (
    Cone,
    MeshPlan,
    PolarBase,
    PolarShell,
    PolarSolidSphere,
    Quality,
    RectBase,
    SlabLayer,
    VoidFill,
    plan_mesh,
)


class MeshPlannerTest(unittest.TestCase):
    def test_solid_sphere_in_vacuum(self):
        plan = plan_mesh(
            [PolarSolidSphere(r=0.05, rho=1.0), VoidFill()],
            PolarBase(s_max=0.2),
        )

        self.assertIsInstance(plan, MeshPlan)
        regions = plan.mesh_kwargs["auto_regions"]
        # [0, 0.05] is solid and [0.05, 0.2] is the trailing void.
        self.assertEqual([region["r_end"] for region in regions], [0.05, 0.2])
        self.assertEqual([region["is_void"] for region in regions], [False, True])
        self.assertTrue(
            all(left["r_end"] < right["r_end"]
                for left, right in zip(regions, regions[1:]))
        )
        self.assertNotIn("grid_theta", plan.mesh_kwargs)
        self.assertNotIn("nr", plan.mesh_kwargs)
        self.assertEqual(
            plan.mesh_kwargs["logical_mesh_2d"],
            "spherical_polar_halfplane",
        )
        self.assertEqual(plan.mesh_kwargs["polar_center_treatment"], "tri_fan")
        self.assertEqual(plan.mesh_kwargs["spherical_polar_s_max"], 0.2)

    def test_three_layer_slab(self):
        plan = plan_mesh(
            [
                SlabLayer(0.0, 0.1, 1.0),
                SlabLayer(0.1, 0.12, 8.9),
                SlabLayer(0.12, 0.2, 0.3),
            ],
            RectBase(r_min=0.0, r_max=0.1, z_min=0.0, z_max=0.3),
        )

        regions = plan.mesh_kwargs["auto_regions"]
        # Three material ends plus the [0.2, 0.3] trailing-void end.
        self.assertEqual(
            [region["r_end"] for region in regions],
            [0.1, 0.12, 0.2, 0.3],
        )
        self.assertEqual(len(regions), 4)
        self.assertEqual(plan.mesh_kwargs["auto_regions_axis"], "z")

    def test_cone_on_shell(self):
        plan = plan_mesh(
            [
                PolarShell(0.04, 0.05, 1.0),
                Cone(theta_half=0.6, n_wall=8, refine_band=0.05),
            ],
            PolarBase(s_max=0.2),
        )

        self.assertIn("auto_regions", plan.mesh_kwargs)
        self.assertEqual(
            plan.mesh_kwargs["grid_theta"]["grading"],
            {"edge_ratio": 0.9999999999999999},
        )
        theta_segments = plan.mesh_kwargs["grid_theta"]["segments"]
        # 0.6 - 0.05 = 0.55 and 0.6 + 0.05 = 0.65 exactly.
        boundaries = [theta_segments[0]["r_start"]] + [
            segment["r_end"] for segment in theta_segments
        ]
        self.assertEqual(boundaries, [0.0, 0.55, 0.6, 0.65, math.pi])
        wall_segments = [
            segment
            for segment in theta_segments
            if segment["r_start"] in (0.55, 0.6)
            and segment["r_end"] in (0.6, 0.65)
        ]
        self.assertEqual([segment["nr"] for segment in wall_segments], [8, 8])
        # Outer counts are round(32*0.55/pi) = 6 and
        # round(32*(pi - 0.65)/pi) = 25, so nz = 6 + 8 + 8 + 25 = 47.
        self.assertEqual(sum(segment["nr"] for segment in theta_segments), 47)
        self.assertNotIn("nz", plan.mesh_kwargs)

    def test_multiblock_capsule_shell(self):
        plan = plan_mesh(
            [PolarShell(0.15, 0.16, 1.0)],
            PolarBase(s_max=0.3, multiblock_r_match=0.1),
        )

        self.assertEqual(
            plan.mesh_kwargs["grid_r"]["grading"],
            {"edge_ratio": 0.9999999999999999},
        )
        segments = plan.mesh_kwargs["grid_r"]["segments"]
        # The shell pins 0.15 and 0.16 inside the shell domain [0.1, 0.3].
        boundaries = [segments[0]["r_start"]] + [
            segment["r_end"] for segment in segments
        ]
        self.assertEqual(boundaries, [0.1, 0.15, 0.16, 0.3])
        self.assertIn(("r", 0.15), plan.report["interfaces_pinned"])
        self.assertIn(("r", 0.16), plan.report["interfaces_pinned"])
        self.assertIn(
            "multiblock_keys_required",
            plan.report["directions"]["r"]["notes"],
        )
        self.assertEqual(
            plan.mesh_kwargs["topology_scheme"],
            "multiblock_cart_core_polar_shell",
        )
        # Three default-count segments give nr = 3 * 16 = 48.
        self.assertEqual(plan.mesh_kwargs["nr"], 48)
        self.assertEqual(
            plan.mesh_kwargs["nr"], sum(segment["nr"] for segment in segments)
        )

    def test_rejections(self):
        with self.assertRaisesRegex(ValueError, "SlabLayer"):
            plan_mesh(
                [SlabLayer(0.0, 0.1, 1.0)],
                PolarBase(s_max=0.2),
            )

        with self.assertRaisesRegex(ValueError, "overlapping"):
            plan_mesh(
                [
                    PolarShell(0.02, 0.06, 1.0),
                    PolarShell(0.05, 0.08, 1.0),
                ],
                PolarBase(s_max=0.2),
            )

    def test_proportional_radial_counts(self):
        plan = plan_mesh(
            [
                PolarShell(100.0, 101.0, 1.0),
                PolarShell(101.0, 102.0, 7.0),
            ],
            PolarBase(
                s_max=102.0,
                center_treatment="annular",
                s_inner=100.0,
            ),
            Quality(n_radial_hint=24),
        )

        self.assertEqual(
            plan.mesh_kwargs["grid_r"]["grading"],
            {"edge_ratio": 0.9999999999999999},
        )
        # M0 = 1 * (101^3 - 100^3) = 30,301.
        # M1 = 7 * (102^3 - 101^3) = 216,349.
        # The proportional counts are round(24*M/[246,650]) = [3, 21].
        counts = [
            segment["nr"] for segment in plan.mesh_kwargs["grid_r"]["segments"]
        ]
        self.assertEqual(counts, [3, 21])
        self.assertEqual(plan.mesh_kwargs["nr"], 24)


if __name__ == "__main__":
    unittest.main()

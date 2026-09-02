#!/usr/bin/env python3
"""Render one drive-asymmetry mode as a four-page PDF report.

Pages: density maps, mesh structure (full domain), mesh structure
(late-time zoom), front-shape visibility. Reuses the page functions of
make_asym_judgment_stack.py, including its Agg flattening of rasterized
pages (PDF mixed-mode renderer defect workaround).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from make_asym_judgment_stack import (
    _flatten_rasterized_page,
    _page_front_shape_visibility,
    _page_mesh_structure,
    _page_rho_maps,
    _placeholder_figure,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-root",
        type=Path,
        required=True,
        help="Directory containing the arm run directory",
    )
    parser.add_argument(
        "--arm", required=True, help="Arm run directory name under arm-root"
    )
    parser.add_argument(
        "--analytic-ratio",
        type=float,
        default=math.nan,
        help="Reference pole/equator drive ratio drawn on the visibility "
        "page (NaN hides the line)",
    )
    parser.add_argument(
        "--analytic-label",
        default="",
        help="Legend label for the reference ratio line",
    )
    parser.add_argument(
        "--out-pdf", type=Path, required=True, help="Output four-page PDF path"
    )
    parser.add_argument(
        "--verdict",
        required=True,
        help="One-line verdict burned into every page title",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.formatter.useoffset": False,
            "savefig.dpi": 150,
        }
    )
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    page_builders = [
        (
            f"{args.arm} density maps",
            lambda: _page_rho_maps(
                args.arm_root, args.arm, args.verdict, page_label="Page 1"
            ),
        ),
        (
            f"{args.arm} mesh structure",
            lambda: _page_mesh_structure(
                args.arm_root,
                args.arm,
                args.verdict,
                page_label="Page 2",
                zoom_late=False,
            ),
        ),
        (
            f"{args.arm} mesh structure (late-time zoom)",
            lambda: _page_mesh_structure(
                args.arm_root,
                args.arm,
                args.verdict,
                page_label="Page 3",
                zoom_late=True,
            ),
        ),
        (
            f"front-shape visibility ({args.arm})",
            lambda: _page_front_shape_visibility(
                args.arm_root,
                args.arm,
                args.analytic_ratio,
                args.analytic_label,
                args.verdict,
                page_label="Page 4",
            ),
        ),
    ]
    metadata = {
        "Title": "TENRYU drive-asymmetry mode report",
        "Author": "TENRYU validation tools",
        "Creator": "make_asym_mode_report.py",
        "CreationDate": dt.datetime(2000, 1, 1),
        "ModDate": dt.datetime(2000, 1, 1),
    }
    with PdfPages(args.out_pdf, metadata=metadata) as pdf:
        for page_number, (page_name, builder) in enumerate(
            page_builders, start=1
        ):
            try:
                figure = builder()
            except (OSError, KeyError, ValueError, IndexError) as error:
                figure = _placeholder_figure(
                    page_number,
                    page_name,
                    args.verdict,
                    f"{type(error).__name__}: {error}",
                )
            if any(
                bool(artist.get_rasterized())
                for artist in figure.findobj()
            ):
                figure = _flatten_rasterized_page(figure)
            pdf.savefig(figure, dpi=300)
            plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

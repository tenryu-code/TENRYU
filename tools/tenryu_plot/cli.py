"""Command-line entry point for tenryu_plot."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tenryu-plot",
        description="Standard 1D post-processing plots for TENRYU outputs.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    from .cmd import compare, convergence, history, laser, profile, spacetime, spectrum, summary

    profile.add_parser(subparsers)
    history.add_parser(subparsers)
    spacetime.add_parser(subparsers)
    summary.add_parser(subparsers)
    compare.add_parser(subparsers)
    convergence.add_parser(subparsers)
    spectrum.add_parser(subparsers)
    laser.add_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError, IndexError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

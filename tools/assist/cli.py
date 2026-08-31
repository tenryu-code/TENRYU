"""Command-line interface for the experimental assistant infrastructure."""

import argparse
import json
import sys
from typing import List, Optional

from tools.assist.config import AssistConfigError, load_config, resolved_summary
from tools.assist.tomlmini import TomlSubsetError


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="assist.py")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--config")
    status_parser.add_argument("--json", action="store_true")

    digest_parser = subparsers.add_parser("digest")
    digest_parser.add_argument("output_dir")
    digest_parser.add_argument("-o", "--output")
    digest_parser.add_argument("--series-points", type=int, default=33)

    zoning_parser = subparsers.add_parser("zoning-report")
    zoning_parser.add_argument("output_dir")
    zoning_parser.add_argument("--kappa", type=float, default=1.0)
    zoning_parser.add_argument("--ablate-margin", type=float, default=1.2)
    zoning_parser.add_argument("--strict", action="store_true")
    zoning_parser.add_argument("-o", "--output")

    promote_parser = subparsers.add_parser("promote-zoning")
    promote_parser.add_argument("output_dir")
    promote_parser.add_argument("--kappa", type=float)
    promote_parser.add_argument("--margin-cells", type=int, default=3)

    lint_parser = subparsers.add_parser("lint-deck")
    lint_parser.add_argument("deck")
    lint_parser.add_argument("--tenryu")
    lint_parser.add_argument("--baseline")
    lint_parser.add_argument("--intent")
    lint_parser.add_argument("-o", "--output")
    lint_parser.add_argument("--keep-tmp", action="store_true")

    generate_parser = subparsers.add_parser("generate-deck")
    generate_parser.add_argument("spec")
    generate_parser.add_argument("--out-deck", required=True)
    generate_parser.add_argument("--tenryu")
    generate_parser.add_argument("--template")
    generate_parser.add_argument("--intent")
    generate_parser.add_argument("--baseline")
    generate_parser.add_argument("--max-iters", type=int, default=10)
    generate_parser.add_argument("--workdir")
    generate_parser.add_argument("--config")

    freeze_parser = subparsers.add_parser("freeze-baseline")
    freeze_parser.add_argument("deck")
    freeze_parser.add_argument("--tenryu")
    freeze_parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)

    if args.subcommand == "status":
        try:
            config = load_config(cli_path=args.config)
        except (AssistConfigError, TomlSubsetError) as error:
            print("assist: config error: {0}".format(error), file=sys.stderr)
            return 2
        print(json.dumps(resolved_summary(config), indent=2, sort_keys=True))
        return 0

    if args.subcommand == "digest":
        from tools.assist.digest import main_digest

        return main_digest(args)

    if args.subcommand == "zoning-report":
        from tools.assist.zoning import main_zoning_report

        return main_zoning_report(args)

    if args.subcommand == "promote-zoning":
        from tools.assist.zoning import main_promote_zoning

        return main_promote_zoning(args)

    if args.subcommand == "lint-deck":
        from tools.assist.deck_lint import main_lint_deck

        return main_lint_deck(args)

    if args.subcommand == "generate-deck":
        from tools.assist.generate import main_generate_deck

        return main_generate_deck(args)

    if args.subcommand == "freeze-baseline":
        from tools.assist.deck_lint import main_freeze_baseline

        return main_freeze_baseline(args)

    return 2

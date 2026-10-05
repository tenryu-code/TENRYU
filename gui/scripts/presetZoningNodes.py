"""Writes test/fixtures/presetZoningNodes.json for test/zoningIntent.test.ts: the solver's mesh of
each laser preset of TENRYU Studio (whose mesh is a recommended zoning_intent) and of each edited
form of test/zoningIntentForms.ts, read from `tenryu validate <deck> --mesh-preview` output. Every
node, or every stride-th node and the last after --stride=N (which applies to the files after it).

    npx vite-node scripts/dumpPresetDecks.ts <out>                  # the preset decks
    tenryu validate <deck> --mesh-preview > <previews>/<id>.out      # each preset and form deck
    python3 scripts/presetZoningNodes.py "<solver revision>" --stride=10 <previews>/<preset>.out ... \
        --stride=1 <previews>/<form>.out ... > test/fixtures/presetZoningNodes.json
"""

import json
import pathlib
import sys

PREFIX = "TENRYU-MESH-PREVIEW: "


def preview_nodes(path: pathlib.Path) -> list[float]:
    for line in path.read_text().splitlines():
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX):])["r_nodes"]
    raise SystemExit(f"{path}: no mesh preview line")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    stride = 1
    meshes = {}
    for arg in sys.argv[2:]:
        if arg.startswith("--stride="):
            stride = int(arg.split("=", 1)[1])
            continue
        path = pathlib.Path(arg)
        nodes = preview_nodes(path)
        meshes[path.stem] = {
            "nCells": len(nodes) - 1,
            "stride": stride,
            "nodes": nodes[::stride],
            "last": nodes[-1],
        }
    json.dump(
        {"generator": "gui/scripts/presetZoningNodes.py", "solver": sys.argv[1], "presets": meshes},
        sys.stdout,
        indent=1,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

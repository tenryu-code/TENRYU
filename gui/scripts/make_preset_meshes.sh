#!/usr/bin/env bash
# Regenerates the recommended meshes of Studio's laser-driven 1D presets (src/core/presetMeshes.ts):
# writes the presets' decks, runs tools/assist recommend-mesh with a solver binary on a server
# (validating every candidate with `tenryu validate --mesh-preview`), and stores the results.
# Usage: gui/scripts/make_preset_meshes.sh <ssh host> <remote TENRYU checkout> <remote tenryu> <binary description>
#   The checkout must hold the same tools/assist and tools/assist/data as the binary was built from
#   (the reference table's digest is compiled into the binary). REMOTE_SETUP, when set, is run on the
#   server before each command (for example the activation of the environment the binary was built in).
set -euo pipefail
HOST="$1"; REPO="$2"; BIN="$3"; LABEL="$4"
GUI="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/tenryu_preset_meshes.XXXXXX")"
cd "$GUI"
npx vite-node scripts/dumpPresetDecks.ts "$WORK"
REMOTE_DIR="$REPO/studio_preset_meshes"
ssh -o BatchMode=yes "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync -a "$WORK/recommend/" "$HOST:$REMOTE_DIR/"
for deck in "$WORK"/recommend/*.py; do
  id="$(basename "$deck" .py)"
  echo "recommend-mesh $id"
  ssh -o BatchMode=yes "$HOST" "${REMOTE_SETUP:+$REMOTE_SETUP && }cd '$REPO' && python3 tools/assist/assist.py recommend-mesh --deck '$REMOTE_DIR/$id.py' --deck-out '$REMOTE_DIR/${id}_out.py' -o '$REMOTE_DIR/$id.json' --tenryu '$BIN'" \
    > "$WORK/recommend/$id.log" 2>&1 || { echo "recommend-mesh failed for $id (log: $WORK/recommend/$id.log)"; exit 2; }
done
rsync -a "$HOST:$REMOTE_DIR/" "$WORK/recommend/"
npx vite-node scripts/writePresetMeshes.ts "$WORK/recommend" "$LABEL"
echo "work directory: $WORK"

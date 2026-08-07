#!/usr/bin/env bash
# TENRYU Studio run wrapper — detached run + JSON status-file contract.
# Usage:  run_detached.sh <run_dir> <tenryu_bin> <deck_path>
# Contract (GUI plan §1/§5 + Addendum 1):
#   - <run_dir>/status.json : the single authority for state/pid/exit_code
#       {"schema":1,"state":"running|finished|failed","pid":N,"exit_code":N|null,
#        "start_epoch":N,"end_epoch":N|null,"host":"...","deck":"...","log":"run.log"}
#       Atomic tmp+mv writes. "running" is written BEFORE the solver starts;
#       the child gates on its existence so the final write can never lose the race.
#   - <run_dir>/run.log     : combined stdout+stderr of `tenryu run`
#   - Solver CWD = <run_dir>, so a deck-relative Output.directory
#     (e.g. "outputs/<case>") lands inside <run_dir> (`tenryu run` has no
#     --output-dir flag; Output.directory resolves against CWD by spec §6.4.8).
#   - stdout of this script : single line = path of status.json
#   - stop                  : kill -TERM -- -<pid>  (pid = process-group leader:
#       setsid on Linux, job-control (set -m) fallback where setsid(1) is absent
#       e.g. macOS; the child traps TERM, forwards it to the solver, then
#       writes final status)
#   - <run_dir> must be absolute and fresh (one directory per run). Paths must not
#     contain double quotes or newlines (GUI-generated names comply).

set -u

write_status() {
  # $1 run_dir  $2 state  $3 pid  $4 exit_code(json)  $5 start_epoch  $6 end_epoch(json)  $7 deck
  local tmp="$1/status.json.tmp"
  cat > "$tmp" <<EOF
{"schema": 1, "state": "$2", "pid": $3, "exit_code": $4, "start_epoch": $5, "end_epoch": $6, "host": "$(hostname)", "deck": "$7", "log": "run.log"}
EOF
  mv -f "$tmp" "$1/status.json"
}

if [ "${1:-}" = "--child" ]; then
  RUN_DIR="$2"; BIN="$3"; DECK="$4"; START_EPOCH="$5"; RESTART="${6:-}"
  # start gate: wait for the parent's "running" status write
  for _ in $(seq 1 100); do
    [ -f "$RUN_DIR/status.json" ] && break
    sleep 0.05
  done
  cd "$RUN_DIR" || exit 10
  if [ -n "$RESTART" ]; then
    "$BIN" run "$DECK" --restart "$RESTART" >> "$RUN_DIR/run.log" 2>&1 &
  else
    "$BIN" run "$DECK" >> "$RUN_DIR/run.log" 2>&1 &
  fi
  TPID=$!
  trap 'kill -TERM "$TPID" 2>/dev/null' TERM INT
  EC=0
  while true; do
    wait "$TPID"
    EC=$?
    kill -0 "$TPID" 2>/dev/null || break
  done
  STATE=finished
  [ "$EC" -ne 0 ] && STATE=failed
  write_status "$RUN_DIR" "$STATE" $$ "$EC" "$START_EPOCH" "$(date +%s)" "$DECK"
  exit 0
fi

if [ $# -lt 3 ]; then
  echo "usage: run_detached.sh <run_dir> <tenryu_bin> <deck_path>" >&2
  exit 2
fi

RUN_DIR="$1"; BIN="$2"; DECK="$3"; RESTART="${4:-}"

case "$RUN_DIR" in
  /*) : ;;
  *) echo "run_dir must be absolute: $RUN_DIR" >&2; exit 5 ;;
esac
mkdir -p "$RUN_DIR" || exit 6
[ -x "$BIN" ] || { echo "tenryu binary not executable: $BIN" >&2; exit 3; }
[ -f "$DECK" ] || { echo "deck not found: $DECK" >&2; exit 4; }

START_EPOCH=$(date +%s)
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

if command -v setsid >/dev/null 2>&1; then
  setsid bash "$SELF" --child "$RUN_DIR" "$BIN" "$DECK" "$START_EPOCH" "$RESTART" < /dev/null >> "$RUN_DIR/run.log" 2>&1 &
  PID=$!
else
  # macOS has no setsid(1). Job control (set -m) gives the background child
  # its own process group (pgid == child pid), preserving the stop contract;
  # a non-interactive parent exit does not HUP the detached group.
  set -m
  bash "$SELF" --child "$RUN_DIR" "$BIN" "$DECK" "$START_EPOCH" "$RESTART" < /dev/null >> "$RUN_DIR/run.log" 2>&1 &
  PID=$!
  set +m
fi

write_status "$RUN_DIR" running "$PID" null "$START_EPOCH" null "$DECK"
echo "$RUN_DIR/status.json"

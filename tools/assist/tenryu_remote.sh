#!/usr/bin/env bash

set -u

HOST="${TENRYU_REMOTE_HOST:-}"
REPO="${TENRYU_REMOTE_REPO:-}"

if [[ -z "$HOST" ]]; then
  echo "assist remote: TENRYU_REMOTE_HOST is required" >&2
  exit 2
fi
if [[ -z "$REPO" ]]; then
  echo "assist remote: TENRYU_REMOTE_REPO is required" >&2
  exit 2
fi

BIN="${TENRYU_REMOTE_BIN:-$REPO/build/tenryu}"
TMPDIR="${TENRYU_REMOTE_TMPDIR:-/tmp}"
SSH_CMD="${TENRYU_REMOTE_SSH:-ssh}"
SCP_CMD="${TENRYU_REMOTE_SCP:-scp}"
RSYNC_CMD="${TENRYU_REMOTE_RSYNC:-rsync}"

RDECK=""
ROUT=""
RDIR=""

cleanup() {
  if [[ -n "$RDECK" ]]; then
    "$SSH_CMD" "$HOST" "rm -f $RDECK" >/dev/null
  fi
  if [[ -n "$ROUT" ]]; then
    "$SSH_CMD" "$HOST" "rm -f $ROUT" >/dev/null
  fi
  if [[ -n "$RDIR" ]]; then
    "$SSH_CMD" "$HOST" "rm -rf $RDIR" >/dev/null
  fi
}

trap cleanup EXIT

guard_argument() {
  local argument="$1"
  if [[ ! "$argument" =~ ^[A-Za-z0-9_./+=:@-]+$ ]]; then
    echo "assist remote: unsupported character in argument: $argument" >&2
    exit 2
  fi
}

make_remote_temp() {
  local template="$1"
  local path
  local status

  path=$("$SSH_CMD" "$HOST" "mktemp $template")
  status=$?
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi
  printf '%s' "$path"
}

upload_deck() {
  local local_deck="$1"
  local status

  RDECK=$(make_remote_temp "$TMPDIR/assist_remote_XXXXXX.py")
  status=$?
  if [[ $status -ne 0 ]]; then
    exit "$status"
  fi
  "$SCP_CMD" "$local_deck" "$HOST:$RDECK" 1>&2
  status=$?
  if [[ $status -ne 0 ]]; then
    echo "assist remote: failed to copy deck to remote host" >&2
    exit 2
  fi
}

if [[ $# -eq 0 ]]; then
  echo "assist remote: unsupported subcommand ''" >&2
  exit 2
fi

subcommand="$1"

case "$subcommand" in
  validate)
    if [[ $# -lt 2 ]]; then
      echo "usage: tenryu_remote.sh validate LOCAL_DECK [extra...]" >&2
      exit 2
    fi
    shift
    local_deck="$1"
    shift
    guard_argument "$local_deck"
    for argument in "$@"; do
      guard_argument "$argument"
    done

    upload_deck "$local_deck"
    command="cd $REPO && $BIN validate $RDECK"
    for argument in "$@"; do
      command="$command $argument"
    done
    "$SSH_CMD" "$HOST" "$command"
    status=$?
    exit "$status"
    ;;

  freeze)
    if [[ $# -ne 4 || "$3" != "-o" ]]; then
      echo "usage: tenryu_remote.sh freeze LOCAL_DECK -o LOCAL_OUT" >&2
      exit 2
    fi
    local_deck="$2"
    local_out="$4"
    guard_argument "$local_deck"
    guard_argument "$local_out"

    upload_deck "$local_deck"
    ROUT=$(make_remote_temp "$TMPDIR/assist_remote_XXXXXX.json")
    status=$?
    if [[ $status -ne 0 ]]; then
      exit "$status"
    fi
    "$SSH_CMD" "$HOST" "cd $REPO && $BIN freeze $RDECK -o $ROUT"
    status=$?
    if [[ $status -ne 0 ]]; then
      exit "$status"
    fi
    "$SCP_CMD" "$HOST:$ROUT" "$local_out" 1>&2
    status=$?
    if [[ $status -ne 0 ]]; then
      echo "assist remote: failed to copy frozen output from remote host" >&2
      exit 2
    fi
    exit 0
    ;;

  run)
    if [[ $# -lt 2 ]]; then
      echo "usage: tenryu_remote.sh run LOCAL_DECK [extra...]" >&2
      exit 2
    fi
    shift
    local_deck="$1"
    shift
    guard_argument "$local_deck"

    local_dir=""
    forwarded_args=""
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == "--output-dir" ]]; then
        if [[ $# -lt 2 ]]; then
          echo "usage: tenryu_remote.sh run LOCAL_DECK [--output-dir LOCAL_DIR] [extra...]" >&2
          exit 2
        fi
        local_dir="$2"
        guard_argument "$local_dir"
        shift 2
      else
        guard_argument "$1"
        forwarded_args="$forwarded_args $1"
        shift
      fi
    done

    upload_deck "$local_deck"
    RDIR=$("$SSH_CMD" "$HOST" "mktemp -d $TMPDIR/assist_remote_out_XXXXXX")
    status=$?
    if [[ $status -ne 0 ]]; then
      exit "$status"
    fi
    command="cd $REPO && $BIN run $RDECK --output-dir $RDIR/out$forwarded_args"
    "$SSH_CMD" "$HOST" "$command"
    status=$?
    if [[ $status -ne 0 ]]; then
      exit "$status"
    fi
    if [[ -n "$local_dir" ]]; then
      if ! mkdir -p "$local_dir"; then
        echo "assist remote: failed to create local output directory" >&2
        exit 2
      fi
      "$RSYNC_CMD" -a "$HOST:$RDIR/out/" "$local_dir/" 1>&2
      status=$?
      if [[ $status -ne 0 ]]; then
        echo "assist remote: failed to copy run output from remote host" >&2
        exit 2
      fi
    fi
    exit 0
    ;;

  --version)
    "$SSH_CMD" "$HOST" "cd $REPO && $BIN --version"
    status=$?
    exit "$status"
    ;;

  *)
    echo "assist remote: unsupported subcommand '$subcommand'" >&2
    exit 2
    ;;
esac

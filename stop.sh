#!/usr/bin/env bash
# Stops the Mars Terrain Intelligence backend and frontend started by start.sh.
#
# Usage: ./stop.sh [--training-wait SECONDS]
#
# 1. The backend receives SIGTERM first. uvicorn then shuts down normally, and the application asks an
#    active training run to stop after its current batch (waiting up to ~20 s) before exiting.
# 2. Anything left in the backend/frontend process groups (npm -> node/vite, a training process that
#    did not stop) is terminated, and killed with SIGKILL if it still does not exit.
#    Completed-epoch checkpoints (checkpoints/<id>/best.pt and last.pt) are kept even after a forced
#    kill, because they are written atomically (temporary file, then rename).
# 3. Anything still listening on ports 8000/5173 that is a python or node process is stopped; other
#    programs on those ports are reported, never touched.
# 4. The PID files in .run/ are removed.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
RUN_DIR="$ROOT/.run"
PORTS="8000 5173"
TRAINING_WAIT=30

while [ $# -gt 0 ]; do
  case "$1" in
    --training-wait) shift; TRAINING_WAIT="${1:?--training-wait needs a value}" ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ -t 1 ]; then
  C_STEP=$'\033[36m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_OFF=$'\033[0m'
else
  C_STEP=""; C_OK=""; C_WARN=""; C_OFF=""
fi
step() { printf '%s==> %s%s\n' "$C_STEP" "$1" "$C_OFF"; }
ok()   { printf '    %s[ok]%s %s\n' "$C_OK" "$C_OFF" "$1"; }
info() { printf '    %s\n' "$1"; }
warn() { printf '    %s[warning]%s %s\n' "$C_WARN" "$C_OFF" "$1"; }

STOPPED=""
note_stopped() { STOPPED="${STOPPED}  - $1"$'\n'; }

alive() { kill -0 "$1" 2>/dev/null; }

group_alive() { kill -0 -- "-$1" 2>/dev/null; }

# Wait until a process (and its process group) has exited, up to N seconds.
wait_gone() {  # pid seconds
  local i=0
  while [ "$i" -lt "$2" ]; do
    if ! alive "$1" && ! group_alive "$1"; then return 0; fi
    sleep 1
    i=$((i + 1))
  done
  return 1
}

stop_group() {  # name pid first-wait-seconds
  local name="$1" pid="$2" wait_s="$3"
  if ! alive "$pid" && ! group_alive "$pid"; then
    info "$name - PID $pid was not running"
    return
  fi
  if alive "$pid"; then
    # Guard against a stale PID file whose number now belongs to an unrelated program.
    local comm
    comm="$(ps -p "$pid" -o comm= 2>/dev/null | sed 's#.*/##')"
    case "$comm" in
      python*|Python*|node*|npm*|sh|bash|nohup) ;;
      *) warn "$name - PID $pid now belongs to '$comm', not to this app; left running."; return ;;
    esac
  fi
  if [ "$name" = "backend" ]; then
    kill -TERM "$pid" 2>/dev/null || true             # graceful: lets the app stop training cleanly
  else
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
  if ! wait_gone "$pid" "$wait_s"; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    if ! wait_gone "$pid" 5; then
      warn "$name did not exit; sending SIGKILL"
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  note_stopped "$name (PID $pid, with child processes)"
  ok "$name stopped (PID $pid)"
}

listening_pids() {  # port
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -t -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | sort -u
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnpH 2>/dev/null | awk -v p="$1" '$4 ~ ("[:.]" p "$")' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u
  fi
}

# ----------------------------------------------------------------------------- 1. PID files
step "Stopping processes started by start.sh"
for name in backend frontend; do
  pid_file="$RUN_DIR/$name.pid"
  if [ ! -f "$pid_file" ]; then
    info "$name - no PID file"
    continue
  fi
  pid="$(tr -dc '0-9' < "$pid_file")"
  if [ -n "$pid" ]; then
    if [ "$name" = "backend" ]; then stop_group "$name" "$pid" "$TRAINING_WAIT"; else stop_group "$name" "$pid" 5; fi
  fi
  rm -f "$pid_file"
done

# ----------------------------------------------------------------------------- 2. leftover listeners
step "Checking ports 8000 and 5173"
for port in $PORTS; do
  pids="$(listening_pids "$port")"
  if [ -z "$pids" ]; then
    ok "port $port is free"
    continue
  fi
  for p in $pids; do
    comm="$(ps -p "$p" -o comm= 2>/dev/null | sed 's#.*/##')"
    case "$comm" in
      python*|Python*|node|node.*)
        kill -TERM "$p" 2>/dev/null || true
        sleep 2
        alive "$p" && kill -KILL "$p" 2>/dev/null
        note_stopped "$comm on port $port (PID $p)"
        ok "stopped $comm (PID $p) listening on port $port"
        ;;
      *)
        warn "port $port is used by '${comm:-unknown}' (PID $p), which was not started by this app; left running."
        ;;
    esac
  done
done

# ----------------------------------------------------------------------------- summary
echo
if [ -n "$STOPPED" ]; then
  printf '%sStopped:%s\n%s' "$C_OK" "$C_OFF" "$STOPPED"
else
  printf '%sNothing was running.%s\n' "$C_OK" "$C_OFF"
fi

#!/usr/bin/env bash
# Starts the Mars Terrain Intelligence backend (port 8000) and frontend (port 5173) in the background.
#
# Usage: ./start.sh [--open] [--timeout SECONDS]
#
# Output goes to .run/backend.log, .run/backend.err.log, .run/frontend.log and .run/frontend.err.log;
# process ids to .run/backend.pid and .run/frontend.pid. Stop with ./stop.sh.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

BACKEND_PORT=8000
FRONTEND_PORT=5173
RUN_DIR="$ROOT/.run"
VENV_PY="$ROOT/.venv/bin/python"
HEALTH_URL="http://127.0.0.1:$BACKEND_PORT/api/health"
FRONTEND_URL="http://127.0.0.1:$FRONTEND_PORT"
TIMEOUT=60
OPEN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --open) OPEN=1 ;;
    --timeout) shift; TIMEOUT="${1:?--timeout needs a value}" ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ -t 1 ]; then
  C_STEP=$'\033[36m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_OFF=$'\033[0m'
else
  C_STEP=""; C_OK=""; C_WARN=""; C_ERR=""; C_OFF=""
fi
step() { printf '%s==> %s%s\n' "$C_STEP" "$1" "$C_OFF"; }
ok()   { printf '    %s[ok]%s %s\n' "$C_OK" "$C_OFF" "$1"; }
warn() { printf '    %s[warning]%s %s\n' "$C_WARN" "$C_OFF" "$1"; }
fail() { printf '\n%s[start failed]%s %s\n' "$C_ERR" "$C_OFF" "$1" >&2; exit 1; }

# True when something listens on the port. Tries lsof, then ss, then a direct connection.
port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 && return 0
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$port\$" && return 0
  fi
  (exec 3<>"/dev/tcp/127.0.0.1/$port") >/dev/null 2>&1 && return 0
  return 1
}

http_ok() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -o /dev/null --max-time 3 "$1" >/dev/null 2>&1
  else
    "$VENV_PY" -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=3)" "$1" >/dev/null 2>&1
  fi
}

wait_http() {  # url pid seconds
  local url="$1" pid="$2" deadline=$((SECONDS + $3))
  while [ "$SECONDS" -lt "$deadline" ]; do
    kill -0 "$pid" 2>/dev/null || return 1
    http_ok "$url" && return 0
    sleep 1
  done
  return 1
}

kill_group() {  # pid: the process was started as its own process group
  kill -TERM -- "-$1" 2>/dev/null || kill -TERM "$1" 2>/dev/null || true
}

show_tail() {
  if [ -s "$1" ]; then
    echo "---- last lines of $1 ----"
    tail -n 25 "$1" | sed 's/^/    /'
  fi
}

# ----------------------------------------------------------------------------- prerequisites
[ -x "$VENV_PY" ] || fail "The Python environment (.venv) is missing. Run ./setup.sh first."
[ -d frontend/node_modules ] || fail "Frontend packages (frontend/node_modules) are missing. Run ./setup.sh first."
command -v npm >/dev/null 2>&1 || fail "npm was not found on PATH."
mkdir -p "$RUN_DIR"

for name in backend frontend; do
  pid_file="$RUN_DIR/$name.pid"
  if [ -f "$pid_file" ]; then
    old_pid="$(tr -dc '0-9' < "$pid_file")"
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
      fail "The $name appears to be running already (PID $old_pid). Run ./stop.sh first."
    fi
    rm -f "$pid_file"   # stale file from an earlier session
  fi
done

for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if port_in_use "$port"; then
    fail "Port $port is already in use by another program. Run ./stop.sh, or find the program with: lsof -iTCP:$port -sTCP:LISTEN"
  fi
done

# Job control gives each background job its own process group (PGID = PID), so stop.sh can stop
# npm together with the node/vite processes it spawns.
set -m

# ----------------------------------------------------------------------------- backend
step "Starting backend on port $BACKEND_PORT"
nohup "$VENV_PY" -m uvicorn backend.app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" \
  >"$RUN_DIR/backend.log" 2>"$RUN_DIR/backend.err.log" </dev/null &
BACKEND_PID=$!
echo "$BACKEND_PID" >"$RUN_DIR/backend.pid"

echo "    waiting for $HEALTH_URL (up to $TIMEOUT s)"
if ! wait_http "$HEALTH_URL" "$BACKEND_PID" "$TIMEOUT"; then
  show_tail "$RUN_DIR/backend.err.log"
  kill_group "$BACKEND_PID"
  rm -f "$RUN_DIR/backend.pid"
  fail "The backend did not become ready. Full logs: $RUN_DIR/backend.log and $RUN_DIR/backend.err.log"
fi
ok "backend ready (PID $BACKEND_PID)"

# ----------------------------------------------------------------------------- frontend
step "Starting frontend on port $FRONTEND_PORT"
cd frontend
nohup npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort \
  >"$RUN_DIR/frontend.log" 2>"$RUN_DIR/frontend.err.log" </dev/null &
FRONTEND_PID=$!
cd "$ROOT"
echo "$FRONTEND_PID" >"$RUN_DIR/frontend.pid"

echo "    waiting for $FRONTEND_URL (up to $TIMEOUT s)"
if ! wait_http "$FRONTEND_URL" "$FRONTEND_PID" "$TIMEOUT"; then
  show_tail "$RUN_DIR/frontend.err.log"
  show_tail "$RUN_DIR/frontend.log"
  kill_group "$FRONTEND_PID"
  rm -f "$RUN_DIR/frontend.pid"
  warn "The backend is still running; stop it with ./stop.sh."
  fail "The frontend did not become ready. Full logs: $RUN_DIR/frontend.log and $RUN_DIR/frontend.err.log"
fi
ok "frontend ready (PID $FRONTEND_PID)"

# ----------------------------------------------------------------------------- done
printf '\n%sMars Terrain Intelligence is running.%s\n' "$C_OK" "$C_OFF"
echo "  Web app:   http://localhost:$FRONTEND_PORT   (or $FRONTEND_URL)"
echo "  API docs:  http://127.0.0.1:$BACKEND_PORT/docs"
echo "  Logs:      $RUN_DIR"
echo "  Stop with: ./stop.sh"

if [ "$OPEN" = 1 ]; then
  if command -v open >/dev/null 2>&1 && [ "$(uname -s)" = "Darwin" ]; then
    open "http://localhost:$FRONTEND_PORT"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1 || true
  else
    warn "Could not find a command to open the browser; open http://localhost:$FRONTEND_PORT manually."
  fi
fi

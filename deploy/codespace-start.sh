#!/usr/bin/env bash
# Start the FULL app in public mode on port 8000 and (in GitHub Codespaces) make the port public.
#   bash deploy/codespace-start.sh
# Works anywhere with the repo set up (./setup.sh); in a Codespace it runs automatically on every start.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .run

[ -x .venv/bin/python ] || ./setup.sh --cpu-only

# Website: build once (or again when the frontend source is newer than the build).
if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src -newer frontend/dist/index.html -print -quit)" ]; then
  echo "==> Building the website"
  (cd frontend && npm run build >/dev/null)
fi

# First start: begin with the trained models, results and test images from deploy/bundle.
if [ ! -f data/mti.sqlite3 ]; then
  echo "==> Copying the trained models and results"
  mkdir -p data checkpoints experiments
  cp -rn deploy/bundle/data/. data/ 2>/dev/null
  cp -rn deploy/bundle/checkpoints/. checkpoints/ 2>/dev/null
  cp -rn deploy/bundle/experiments/. experiments/ 2>/dev/null
fi

TOKEN_FILE=.mti_admin_token
[ -s "$TOKEN_FILE" ] || (umask 077; openssl rand -base64 18 | tr -d '/+=' > "$TOKEN_FILE")

pkill -f "uvicorn backend.app.main:app --host 0.0.0.0 --port 8000" 2>/dev/null || true
MTI_PUBLIC_MODE=true MTI_ADMIN_TOKEN="$(cat "$TOKEN_FILE")" MTI_FRONTEND_DIST="$PWD/frontend/dist" MTI_CORS_ORIGINS="" \
  nohup .venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 > .run/public.log 2>&1 &

echo "==> Waiting for the app"
for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break; sleep 2; done
curl -fsS http://127.0.0.1:8000/api/health >/dev/null || { echo "The app did not start; see .run/public.log"; exit 1; }

URL="http://localhost:8000"
if [ -n "${CODESPACE_NAME:-}" ]; then
  URL="https://${CODESPACE_NAME}-8000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
  gh codespace ports visibility 8000:public -c "$CODESPACE_NAME" >/dev/null 2>&1 \
    && echo "==> Port 8000 is public" \
    || echo "==> Make the link public: PORTS tab -> right-click port 8000 -> Port Visibility -> Public"
fi
cat <<MSG

Mars Terrain Intelligence (full app) is running.
  Public link:  $URL
  Admin token:  cat $TOKEN_FILE   (enter it on the website under Settings -> Admin token)
  Log:          .run/public.log
MSG

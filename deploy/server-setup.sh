#!/usr/bin/env bash
# One-command setup of the FULL app on a fresh Ubuntu server (e.g. Oracle Cloud "Always Free" VM).
#
#   curl -fsSL https://raw.githubusercontent.com/aarushagrawal267-maker/AI4Mars/main/deploy/server-setup.sh | bash
#   (or: bash deploy/server-setup.sh [admin-password])
#
# Installs Docker, downloads the repository, builds deploy/render.Dockerfile (website + API + trained models),
# and runs it on port 80 in public demo mode with automatic restart. Data, trained models and experiments are kept
# in Docker volumes, so they survive restarts and updates. Run it again to update to the latest code.
set -euo pipefail

ADMIN_TOKEN="${1:-${MTI_ADMIN_TOKEN:-}}"
REPO="https://github.com/aarushagrawal267-maker/AI4Mars.git"
DIR="$HOME/AI4Mars"
PORT=80

say() { printf '\n==> %s\n' "$*"; }

say "Installing Docker and Git"
sudo apt-get update -y
sudo apt-get install -y docker.io git curl openssl
sudo systemctl enable --now docker

say "Opening port $PORT in the server's own firewall (Oracle Ubuntu images block it by default)"
if sudo iptables -C INPUT -p tcp --dport "$PORT" -j ACCEPT 2>/dev/null; then
  echo "    already open"
else
  sudo iptables -I INPUT 1 -p tcp --dport "$PORT" -j ACCEPT
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y iptables-persistent >/dev/null 2>&1 || true
  sudo netfilter-persistent save >/dev/null 2>&1 || true
fi

say "Downloading the latest code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else git clone --depth 1 "$REPO" "$DIR"; fi

say "Building the app image (first time: 10-20 minutes, it installs PyTorch)"
sudo docker build -f "$DIR/deploy/render.Dockerfile" -t mars-terrain-intelligence "$DIR"

TOKEN_FILE="$HOME/.mti_admin_token"
if [ -z "$ADMIN_TOKEN" ]; then
  if [ -f "$TOKEN_FILE" ]; then ADMIN_TOKEN="$(cat "$TOKEN_FILE")"; else ADMIN_TOKEN="$(openssl rand -base64 18 | tr -d '/+=')"; fi
fi
umask 077; printf '%s\n' "$ADMIN_TOKEN" > "$TOKEN_FILE"

say "Starting the app"
sudo docker rm -f mti >/dev/null 2>&1 || true
# Named volumes start as a copy of the image's folders (trained models, results, test images) and then persist.
sudo docker run -d --name mti --restart unless-stopped -p "$PORT":10000 \
  -e MTI_PUBLIC_MODE=true -e MTI_ADMIN_TOKEN="$ADMIN_TOKEN" -e MTI_CORS_ORIGINS="" \
  -v mti-data:/app/data -v mti-checkpoints:/app/checkpoints -v mti-experiments:/app/experiments \
  mars-terrain-intelligence

say "Waiting for the app to answer"
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null || { echo "The app did not start; see: sudo docker logs mti"; exit 1; }

IP="$(curl -fsS https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}')"
cat <<EOF

Mars Terrain Intelligence is running.
  Public URL:   http://$IP
  Admin token:  saved in $TOKEN_FILE  (show it with: cat $TOKEN_FILE)
                enter it on the website under Settings -> Admin token to unlock training and evaluation
  Logs:         sudo docker logs -f mti
  Update later: bash $DIR/deploy/server-setup.sh

If the URL does not open from your computer, allow TCP port $PORT in the cloud provider's firewall
(Oracle: Networking -> Virtual cloud networks -> your VCN -> Security Lists -> Add Ingress Rule, source 0.0.0.0/0, port $PORT).
EOF

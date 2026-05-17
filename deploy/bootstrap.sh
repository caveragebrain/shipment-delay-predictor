#!/usr/bin/env bash
# One-time droplet bootstrap. Run as user `autoforward` (NOPASSWD sudo).
# Assumes the repo has already been cloned to /opt/shipment-delay-predictor.
set -euo pipefail

APP_DIR=/opt/shipment-delay-predictor
cd "$APP_DIR"

sudo apt-get update
sudo apt-get install -y python3.12-venv libomp-dev

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# .env must already exist with GEMINI_API_KEY
test -f .env || { echo "ERROR: .env missing. Create it with GEMINI_API_KEY=..."; exit 1; }

sudo cp deploy/shipment-delay.service /etc/systemd/system/shipment-delay.service
sudo systemctl daemon-reload
sudo systemctl enable --now shipment-delay.service

# Caddy: append vhost block if not present
if ! sudo grep -q "shipment.autoforward.me" /etc/caddy/Caddyfile; then
    sudo bash -c "cat $APP_DIR/deploy/Caddyfile.snippet >> /etc/caddy/Caddyfile"
    sudo systemctl reload caddy
fi

echo "Bootstrap complete. Health:"
sleep 2
curl -fsS http://127.0.0.1:8001/health

#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# restart_server.sh
# Kill any running scanner server, clear caches, restart.
#
# Usage:
#   cd ~/card-scanner
#   ./restart_server.sh              # Jetson (production)
#   ./restart_server.sh --mac        # Mac (LAN + self-signed SSL)
#   ./restart_server.sh --mock       # Mac (localhost, mock camera)
#   ./restart_server.sh --deploy     # Also copy new files from ~/Downloads
# ──────────────────────────────────────────────────────────
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_DIR}/venv"
CERTS_DIR="${REPO_DIR}/certs"

# ── Parse args ──
MAC_MODE=false
MOCK_MODE=false
DEPLOY=false
for arg in "$@"; do
  case "$arg" in
    --mac)    MAC_MODE=true ;;
    --mock)   MOCK_MODE=true ;;
    --deploy) DEPLOY=true ;;
  esac
done

echo "═══════════════════════════════════════════════════"
echo "  Card Scanner — Server Restart"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Kill existing server ──
echo "[1/5] Stopping existing server..."
pkill -f "python.*server.py" 2>/dev/null && echo "  Killed running server" || echo "  No server running"
sleep 1

# ── Step 2: Deploy updated files (optional) ──
if [ "$DEPLOY" = true ]; then
  echo "[2/5] Deploying updated files..."
  DOWNLOADS="${HOME}/Downloads"
  deployed=0

  for f in scanner.py ocr.py database.py; do
    if [ -f "${DOWNLOADS}/${f}" ]; then
      cp "${DOWNLOADS}/${f}" "${REPO_DIR}/${f}"
      echo "  Deployed: ${f}"
      deployed=$((deployed + 1))
    fi
  done

  if [ $deployed -eq 0 ]; then
    echo "  No updated files found in ${DOWNLOADS}/"
    echo "  (Looking for: scanner.py, ocr.py, database.py)"
  else
    echo "  ${deployed} file(s) deployed"
  fi
else
  echo "[2/5] Skipping deploy (use --deploy to copy new files from ~/Downloads)"
fi

# ── Step 3: Clear __pycache__ ──
echo "[3/5] Clearing Python cache..."
find "${REPO_DIR}" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${REPO_DIR}" -name "*.pyc" -delete 2>/dev/null || true
echo "  Cache cleared"

# ── Step 4: Activate venv + verify deps ──
echo "[4/5] Activating environment..."
if [ -d "${VENV_DIR}" ]; then
  source "${VENV_DIR}/bin/activate"
  echo "  venv activated"
else
  echo "  WARNING: No venv at ${VENV_DIR}, using system Python"
fi

python3 -c "import fastapi, uvicorn, cv2, easyocr" 2>/dev/null || {
  echo "  ERROR: Missing dependencies. Run:"
  echo "    pip install -r requirements.txt"
  exit 1
}
echo "  Dependencies OK"

# ── Step 5: Start server ──
echo "[5/5] Starting server..."
echo ""

if [ "$MOCK_MODE" = true ]; then
  echo "  Mode:   MOCK (localhost only, mock camera)"
  echo "  URL:    http://127.0.0.1:8080/mobile"
  echo ""
  exec python3 "${REPO_DIR}/server.py" \
    --camera-mode mock \
    --log-level info

elif [ "$MAC_MODE" = true ]; then
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "unknown")

  # Generate self-signed cert if needed (for HTTPS camera access)
  if [ ! -f "${CERTS_DIR}/cert.pem" ]; then
    echo "  Generating self-signed SSL certificate..."
    mkdir -p "${CERTS_DIR}"
    openssl req -x509 -newkey rsa:2048 -nodes \
      -keyout "${CERTS_DIR}/key.pem" \
      -out "${CERTS_DIR}/cert.pem" \
      -days 365 \
      -subj "/CN=${LAN_IP}" \
      -addext "subjectAltName=IP:${LAN_IP},IP:127.0.0.1" \
      2>/dev/null
    echo "  Certificate created for ${LAN_IP}"
  fi

  echo "  Mode:   MAC (LAN + HTTPS)"
  echo "  URL:    https://${LAN_IP}:8080/mobile"
  echo ""
  echo "  ┌──────────────────────────────────────────────┐"
  echo "  │  On your phone, open:                        │"
  echo "  │  https://${LAN_IP}:8080/mobile          │"
  echo "  │                                              │"
  echo "  │  First time: tap Advanced → Proceed          │"
  echo "  │  (self-signed cert warning is expected)      │"
  echo "  └──────────────────────────────────────────────┘"
  echo ""
  exec python3 "${REPO_DIR}/server.py" \
    --host 0.0.0.0 \
    --port 8080 \
    --ssl-certfile "${CERTS_DIR}/cert.pem" \
    --ssl-keyfile "${CERTS_DIR}/key.pem" \
    --log-level info

else
  # Jetson production mode
  LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  LAN_IP="${LAN_IP:-<jetson-ip>}"

  echo "  Mode:   PRODUCTION (Jetson, LAN-accessible)"
  echo "  URL:    http://${LAN_IP}:8080/mobile"
  echo ""
  exec python3 "${REPO_DIR}/server.py" \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level info
fi

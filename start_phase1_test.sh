#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────
# start_phase1_test.sh
# Deploys updated files and starts the scanner server
# with the mobile UI accessible on the local network.
#
# Usage:
#   # On the Jetson (production, LAN-accessible):
#   cd ~/card-scanner
#   bash start_phase1_test.sh
#
#   # On Mac (dev, mock camera):
#   cd ~/card-scanner
#   bash start_phase1_test.sh --mock
# ──────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}"
VENV_DIR="${REPO_DIR}/venv"

# ── Parse args ──
MOCK_MODE=false
MAC_MODE=false
for arg in "$@"; do
  case "$arg" in
    --mock) MOCK_MODE=true ;;
    --mac)  MAC_MODE=true ;;
  esac
done

echo "═══════════════════════════════════════════════════"
echo "  Card Scanner — Phase 1 Test Server"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Check we're in the right place ──
if [ ! -f "${REPO_DIR}/scanner.py" ]; then
  echo "ERROR: scanner.py not found in ${REPO_DIR}"
  echo "Run this script from the card-scanner repo root."
  exit 1
fi

# ── Activate venv ──
if [ -d "${VENV_DIR}" ]; then
  echo "[1/4] Activating virtual environment..."
  source "${VENV_DIR}/bin/activate"
else
  echo "WARNING: No venv found at ${VENV_DIR}"
  echo "         Using system Python. If imports fail, create a venv first."
fi

# ── Verify key dependencies ──
echo "[2/4] Checking dependencies..."
python3 -c "import fastapi, uvicorn, cv2, easyocr, imagehash" 2>/dev/null || {
  echo "ERROR: Missing dependencies. Install with:"
  echo "  pip install fastapi uvicorn opencv-python easyocr imagehash pillow"
  exit 1
}
echo "  All dependencies OK"

# ── Ensure test directory exists ──
echo "[3/4] Preparing test data directory..."
mkdir -p "${REPO_DIR}/data/test_phone"
echo "  data/test_phone/ ready"

# ── Start server ──
echo "[4/4] Starting server..."
echo ""

if [ "$MAC_MODE" = true ]; then
  # Mac dev mode: mock camera (no CSI), but LAN-accessible for phone uploads
  # HTTPS required for getUserMedia (phone camera access)
  LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "<mac-ip>")

  # ── Generate self-signed cert if not present ──
  CERT_DIR="${REPO_DIR}/certs"
  CERT_FILE="${CERT_DIR}/localhost.pem"
  KEY_FILE="${CERT_DIR}/localhost-key.pem"

  if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "  Generating self-signed SSL certificate..."
    mkdir -p "$CERT_DIR"
    openssl req -x509 -newkey rsa:2048 -nodes \
      -keyout "$KEY_FILE" \
      -out "$CERT_FILE" \
      -days 365 \
      -subj "/CN=card-scanner" \
      -addext "subjectAltName=IP:${LAN_IP},IP:127.0.0.1,DNS:localhost" \
      2>/dev/null
    echo "  Certificate created at ${CERT_DIR}/"
    echo ""
    echo "  ⚠  First time: your phone will show a security warning."
    echo "     Tap 'Advanced' → 'Proceed' to accept the self-signed cert."
    echo ""
  fi

  echo "  Mode:     MAC (LAN-accessible, HTTPS)"
  echo "  Camera:   mock (phone uploads via /scan/upload)"
  echo "  URL:      https://${LAN_IP}:8080/mobile"
  echo ""
  echo "  ┌─────────────────────────────────────────┐"
  echo "  │  Open this URL on your phone:            │"
  echo "  │  https://${LAN_IP}:8080/mobile      │"
  echo "  │                                          │"
  echo "  │  1. Accept the security warning          │"
  echo "  │  2. Tap 'Test' to enable Test Mode       │"
  echo "  │  3. Set condition (Bare / Sleeve)        │"
  echo "  │  4. Scan cards — data saves auto         │"
  echo "  │  5. Ctrl+C when done, then run:          │"
  echo "  │     python3 test_phone.py                │"
  echo "  └─────────────────────────────────────────┘"
  echo ""
  python3 "${REPO_DIR}/server.py" \
    --host 0.0.0.0 \
    --port 8080 \
    --camera-mode mock \
    --ssl-certfile "$CERT_FILE" \
    --ssl-keyfile "$KEY_FILE" \
    --log-level info

elif [ "$MOCK_MODE" = true ]; then
  echo "  Mode:     MOCK (localhost only)"
  echo "  Camera:   mock images"
  echo "  URL:      http://127.0.0.1:8080/mobile"
  echo ""
  python3 "${REPO_DIR}/server.py" \
    --camera-mode mock \
    --log-level info
else
  # Get the Jetson's LAN IP for the QR code / URL
  LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  LAN_IP="${LAN_IP:-<jetson-ip>}"

  echo "  Mode:     PRODUCTION (LAN-accessible)"
  echo "  Camera:   CSI (auto-detect)"
  echo "  URL:      http://${LAN_IP}:8080/mobile"
  echo ""
  echo "  ┌─────────────────────────────────────────┐"
  echo "  │  Open this URL on your phone:            │"
  echo "  │  http://${LAN_IP}:8080/mobile       │"
  echo "  │                                          │"
  echo "  │  1. Tap 'Test' to enable Test Mode       │"
  echo "  │  2. Set condition (Bare / Sleeve)        │"
  echo "  │  3. Scan cards — data saves auto         │"
  echo "  │  4. Ctrl+C when done, then run:          │"
  echo "  │     python3 test_phone.py                │"
  echo "  └─────────────────────────────────────────┘"
  echo ""
  python3 "${REPO_DIR}/server.py" \
    --host 0.0.0.0 \
    --port 8080 \
    --log-level info
fi

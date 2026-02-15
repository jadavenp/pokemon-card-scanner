#!/usr/bin/env bash
# restart_server.sh — Kill existing server and relaunch
set -euo pipefail

cd ~/card-scanner

# Kill any running server on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null && echo "Killed existing server" || echo "No existing server"

sleep 1

# Relaunch
./start_phase1_test.sh --mac

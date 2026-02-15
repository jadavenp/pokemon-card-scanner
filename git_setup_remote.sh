#!/bin/bash
# git_setup_remote.sh — One-time setup to connect and push to GitHub
# Usage: cd ~/card-scanner && ./git_setup_remote.sh

set -euo pipefail

REPO_DIR="$HOME/card-scanner"
REMOTE_URL="git@github.com:jadavenp/pokemon-card-scanner.git"

cd "$REPO_DIR" 2>/dev/null || {
    echo "ERROR: Directory $REPO_DIR not found."
    exit 1
}

echo "================================================"
echo "  Card Scanner — GitHub Setup"
echo "================================================"
echo ""

# ── Step 1: Check git identity ─────────────────────────────────────
echo "[1/6] Checking git identity..."
if ! git config user.name > /dev/null 2>&1; then
    echo "  No git user.name set. Setting now..."
    read -rp "  Your name: " uname
    git config --global user.name "$uname"
fi
if ! git config user.email > /dev/null 2>&1; then
    git config --global user.email "JamesRyanDavenport@gmail.com"
fi
echo "  Identity: $(git config user.name) <$(git config user.email)>"
echo ""

# ── Step 2: Initialize repo if needed ──────────────────────────────
echo "[2/6] Initializing repo..."
if [ -d .git ]; then
    echo "  Already initialized."
else
    git init
    echo "  Initialized."
fi
echo ""

# ── Step 3: Create .gitignore ──────────────────────────────────────
echo "[3/6] Creating .gitignore..."
cat > .gitignore << 'GITIGNORE'
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# Data files (large — not suitable for git)
data/Ref Images/
data/hash_db.pkl
data/card_index.json
data/tcgplayer_id_map.json

# Test captures (raw images)
data/test_phone/*.jpg
data/test_phone/*.jpeg
data/test_phone/*.png
# DO commit the manifest for test history
!data/test_phone/manifest.json

# SSL certs (auto-generated, machine-specific)
certs/

# OS junk
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo
GITIGNORE
echo "  .gitignore created."
echo ""

# ── Step 4: Stage all tracked files ────────────────────────────────
echo "[4/6] Staging files..."
git add .
echo "  Files staged:"
git diff --cached --name-status | sed 's/^/    /'
echo ""

# ── Step 5: Initial commit ─────────────────────────────────────────
echo "[5/6] Creating initial commit..."
if git log --oneline -1 > /dev/null 2>&1; then
    echo "  Commits already exist. Skipping initial commit."
    echo "  Latest: $(git log --oneline -1)"
else
    git commit -m "Initial commit — card scanner with Phase 1 test infrastructure"
    echo "  Committed."
fi
echo ""

# ── Step 6: Set remote and push ────────────────────────────────────
echo "[6/6] Connecting to GitHub and pushing..."

if git remote get-url origin > /dev/null 2>&1; then
    current=$(git remote get-url origin)
    if [ "$current" != "$REMOTE_URL" ]; then
        echo "  Updating remote from $current"
        git remote set-url origin "$REMOTE_URL"
    fi
    echo "  Remote: $REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
    echo "  Remote added: $REMOTE_URL"
fi

# Ensure we're on main branch
branch=$(git branch --show-current)
if [ "$branch" != "main" ]; then
    git branch -M main
    echo "  Renamed branch to main."
fi

echo ""
echo "  Pushing to origin/main..."
git push -u origin main

echo ""
echo "================================================"
echo "  Done! Repo live at:"
echo "  https://github.com/jadavenp/pokemon-card-scanner"
echo "================================================"
echo ""
echo "From now on, your daily workflow:"
echo "  ./git_manager.sh status"
echo "  ./git_manager.sh add <files>"
echo "  ./git_manager.sh commit \"message\""
echo "  git push"

#!/bin/bash
# git_manager.sh — Git helper for card-scanner project
# Usage: ./git_manager.sh [command] [args...]
#
# Commands:
#   init          Initialize repo + create .gitignore (run once)
#   status        Show working tree status (staged, modified, untracked)
#   diff [file]   Show unstaged changes (optionally for a specific file)
#   add <paths>   Stage files for commit (supports globs, "." for all)
#   reset <paths> Unstage files (undo add)
#   commit <msg>  Commit staged changes with message
#   log [n]       Show last n commits (default: 10)
#   stash         Stash uncommitted changes
#   stash-pop     Restore stashed changes
#   help          Show this help
#
# Examples:
#   ./git_manager.sh init
#   ./git_manager.sh status
#   ./git_manager.sh add scanner.py server.py
#   ./git_manager.sh add .
#   ./git_manager.sh commit "Fix landscape output bug in mobile scanner"
#   ./git_manager.sh diff scanner_mobile.html
#   ./git_manager.sh log 5

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
REPO_DIR="$HOME/card-scanner"
# ────────────────────────────────────────────────────────────────────

cd "$REPO_DIR" 2>/dev/null || {
    echo "ERROR: Directory $REPO_DIR not found."
    echo "Update REPO_DIR at the top of this script if your repo is elsewhere."
    exit 1
}

cmd="${1:-help}"
shift 2>/dev/null || true

case "$cmd" in

    # ── Initialize repo + gitignore ─────────────────────────────────
    init)
        if [ -d .git ]; then
            echo "Repo already initialized at $REPO_DIR"
            echo "Current branch: $(git branch --show-current 2>/dev/null || echo 'no commits yet')"
            git log --oneline -3 2>/dev/null || echo "(no commits yet)"
        else
            git init
            echo "Initialized new git repo at $REPO_DIR"
        fi

        # Create/update .gitignore
        cat > .gitignore << 'GITIGNORE'
# Python
venv/
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# Data files (large — tracked separately or not at all)
data/Ref Images/
data/hash_db.pkl
data/card_index.json
data/tcgplayer_id_map.json

# Test captures (don't commit raw test images)
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

        echo ""
        echo ".gitignore created. Excluded:"
        echo "  - venv/, __pycache__"
        echo "  - data/Ref Images/, hash_db.pkl, card_index.json (~73MB+)"
        echo "  - data/test_phone images (manifest.json IS tracked)"
        echo "  - certs/ (auto-generated)"
        echo ""
        echo "Review with: ./git_manager.sh status"
        echo "Then stage:  ./git_manager.sh add ."
        echo "And commit:  ./git_manager.sh commit \"Initial commit\""
        ;;

    # ── Status ──────────────────────────────────────────────────────
    status)
        echo "=== Git Status: $REPO_DIR ==="
        echo ""
        git status --short --branch
        echo ""

        # Count summary
        staged=$(git diff --cached --name-only 2>/dev/null | wc -l)
        modified=$(git diff --name-only 2>/dev/null | wc -l)
        untracked=$(git ls-files --others --exclude-standard 2>/dev/null | wc -l)
        echo "Summary: ${staged} staged | ${modified} modified | ${untracked} untracked"
        ;;

    # ── Diff ────────────────────────────────────────────────────────
    diff)
        if [ $# -gt 0 ]; then
            git diff "$@"
        else
            git diff
        fi
        ;;

    # ── Add (stage files) ──────────────────────────────────────────
    add)
        if [ $# -eq 0 ]; then
            echo "Usage: ./git_manager.sh add <file1> [file2] ... or add ."
            exit 1
        fi
        git add "$@"
        echo "Staged:"
        git diff --cached --name-status
        ;;

    # ── Reset (unstage files) ──────────────────────────────────────
    reset)
        if [ $# -eq 0 ]; then
            echo "Usage: ./git_manager.sh reset <file1> [file2] ..."
            exit 1
        fi
        git reset HEAD "$@" 2>/dev/null || git rm --cached "$@" 2>/dev/null
        echo "Unstaged: $*"
        ;;

    # ── Commit ─────────────────────────────────────────────────────
    commit)
        msg="${1:-}"
        if [ -z "$msg" ]; then
            echo "Usage: ./git_manager.sh commit \"Your commit message\""
            exit 1
        fi

        # Safety check: show what will be committed
        staged=$(git diff --cached --name-only 2>/dev/null | wc -l)
        if [ "$staged" -eq 0 ]; then
            echo "Nothing staged. Use './git_manager.sh add <files>' first."
            exit 1
        fi

        echo "Committing $staged file(s):"
        git diff --cached --name-status
        echo ""
        git commit -m "$msg"
        ;;

    # ── Log ─────────────────────────────────────────────────────────
    log)
        count="${1:-10}"
        git log --oneline --graph --decorate -"$count"
        ;;

    # ── Stash ───────────────────────────────────────────────────────
    stash)
        git stash push -m "stash-$(date +%Y%m%d-%H%M%S)"
        echo "Changes stashed."
        ;;

    stash-pop)
        git stash pop
        echo "Stash restored."
        ;;

    # ── Help ────────────────────────────────────────────────────────
    help|*)
        echo "git_manager.sh — Git helper for card-scanner"
        echo ""
        echo "Commands:"
        echo "  init          Initialize repo + create .gitignore"
        echo "  status        Show working tree status"
        echo "  diff [file]   Show unstaged changes"
        echo "  add <paths>   Stage files (use '.' for all)"
        echo "  reset <paths> Unstage files"
        echo "  commit <msg>  Commit staged changes"
        echo "  log [n]       Show last n commits (default 10)"
        echo "  stash         Stash uncommitted changes"
        echo "  stash-pop     Restore stashed changes"
        echo "  help          This help"
        echo ""
        echo "Workflow:"
        echo "  1. ./git_manager.sh init          # first time only"
        echo "  2. ./git_manager.sh status         # see what changed"
        echo "  3. ./git_manager.sh add <files>    # stage specific files"
        echo "  4. ./git_manager.sh commit \"msg\"   # commit with message"
        ;;
esac

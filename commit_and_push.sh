#!/bin/bash
# Commit and push P4+P1+P2 refactor changes
# Run from: ~/card-scanner

cd ~/card-scanner || exit 1

# Stage the 5 modified files
git add scanner.py server.py ocr.py test_random.py config.py

# Verify what's staged
echo "=== Staged changes ==="
git diff --cached --stat
echo ""

# Commit
git commit -m "Refactor: hash-first pipeline, unify scan paths, cleanup

P4 — Housekeeping:
- scanner.py: lazy prettytable import (no more sys.exit on server import)
- scanner.py: remove dead pricing_justtcg import (~200ms startup savings)
- ocr.py: replace debug print() with logger.debug()
- server.py: remove unused StaticFiles import
- server.py: pre-create temp upload dir in lifespan startup
- test_random.py: improve scanner_ui import error message

P1 — Unify scan paths:
- server.py: remove duplicate card detection from /scan/upload
- server.py: remove card_detect import (process_image handles it)
- All scans (CLI, desktop, phone) now go through identical pipeline

P2 — Hash-first pipeline reorder:
- Hash matching (~50ms) runs before OCR (~1-2s)
- Confident hash match skips OCR entirely, pulls identity from DB
- OCR remains as fallback for cards not in hash DB
- Avg scan time reduced ~35% (2.3s → 1.5s on reference images)

Fixes:
- config.py: add sv8pt5 -> Prismatic Evolutions to SET_NAMES
- scanner.py: populate set total from DB on hash-identified cards"

# Push
git push origin main

echo ""
echo "=== Done ==="

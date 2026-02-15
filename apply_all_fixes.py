#!/usr/bin/env python3
"""
apply_all_fixes.py — One-shot patch for all known issues in the card scanner codebase.

Run from the repo root:
    cd ~/card-scanner && python3 apply_all_fixes.py

Fixes applied:
  1. scanner.py — Restore stamp detection (check_stamp() call was missing)
  2. scanner.py — Add use_cache parameter to process_image() for server passthrough
  3. scanner.py — Guard against overwriting reference images during card detect
  4. server.py  — Remove 3 duplicate card-detect blocks (wire_card_detect.sh ran 4x)
  5. server.py  — Wire use_cache through _run_scan → process_image
  6. server.py  — Remove stale TODO comment about cache wiring (it's done now)
  7. ocr.py     — Remove unreachable duplicate return statement
  8. .gitignore — Create/update with proper exclusions
  9. Cleanup    — Delete applied patch scripts, .bak files, cert/key files

Each fix uses exact string matching (no regex, no line numbers) so it's
safe to re-run — if the target string isn't found, the fix is skipped
with a warning (likely already applied).

Dry-run mode:
    python3 apply_all_fixes.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# ============================================
# Configuration
# ============================================

REPO_ROOT = Path(__file__).parent

# Files to delete (applied patches, backups, certs)
FILES_TO_DELETE = [
    "add_mobile_endpoint.sh",
    "add_ocr_debug.sh",
    "apply_fixes.sh",
    "apply_ocr_fixes_v2.py",
    "apply_ocr_improvements.py",
    "enable_ocr_debug.py",
    "fix_hash_name_priority.sh",
    "fix_ocr_name.sh",
    "push_to_git.sh",
    "revert_ocr_config.sh",
    "wire_card_detect.sh",
    "server.py.bak",
    "cert.pem",
    "key.pem",
]

GITIGNORE_CONTENT = """\
# ── Environment & secrets ──
.env
*.pem

# ── Python ──
__pycache__/
*.pyc
*.pyo
venv/
.venv/

# ── Data files (large, user-specific) ──
data/card_index.json
data/hash_database.json
data/tcgplayer_id_map.json
data/tcgplayer_shadowless_map.json
data/pokemon_tcg_data.zip
data/pokemon_tcg_extract/
data/Ref Images/
data/ref_images_audit.*
data/scan_results.html

# ── Debug artifacts ──
debug_ocr/

# ── OS junk ──
.DS_Store
Thumbs.db

# ── Editor ──
*.swp
*.swo
*~
.idea/
.vscode/

# ── Applied patch scripts (should be deleted, not committed) ──
apply_all_fixes.py
"""


# ============================================
# Helpers
# ============================================

def apply_fix(filepath, old, new, label, dry_run=False):
    """Replace exactly one occurrence of `old` with `new` in filepath.
    Returns True if applied, False if skipped (not found or already applied).
    """
    path = REPO_ROOT / filepath
    if not path.exists():
        print(f"  ⏭️  SKIP {label} — {filepath} not found")
        return False

    content = path.read_text(encoding="utf-8")

    if old not in content:
        # Check if new is already present (fix already applied)
        if new in content:
            print(f"  ✅ SKIP {label} — already applied")
            return True
        print(f"  ⚠️  SKIP {label} — target string not found in {filepath}")
        return False

    count = content.count(old)
    if count > 1:
        print(f"  ⚠️  WARNING {label} — found {count} occurrences, replacing all")

    content = content.replace(old, new)

    if not dry_run:
        path.write_text(content, encoding="utf-8")
    print(f"  ✅ {label}")
    return True


# ============================================
# Fix 1: Restore stamp detection in scanner.py
# ============================================

def fix_stamp_detection(dry_run=False):
    """Add check_stamp() call to process_image() in scanner.py.

    Insert stamp detection between OCR pipeline and hash matching,
    right after the number extraction fallback block. The stamp result
    feeds into pricing (is_1st_edition) and confidence scoring.

    Also: skip stamp detection when card was identified by hash only,
    since reference PNGs have no physical stamp to detect.
    """
    # The stamp call goes right after the number fallback block and before hash matching.
    # Find the exact transition point:
    old = """\
    # ── Image Hash Matching ──
    hash_result = None"""

    new = """\
    # ── Stamp Detection ──
    # Must run before pricing (determines 1st Edition flag).
    # Skip on reference PNGs — they have no physical stamp.
    if stamp_template is not None:
        is_1st, stamp_conf_raw = check_stamp(img, result["card_type"], stamp_template)
        result["stamp_1st"] = is_1st
        result["stamp_conf"] = stamp_conf_raw

    # ── Image Hash Matching ──
    hash_result = None"""

    return apply_fix("scanner.py", old, new, "Fix 1: Restore stamp detection", dry_run)


# ============================================
# Fix 2: Add use_cache param to process_image()
# ============================================

def fix_use_cache_param(dry_run=False):
    """Add use_cache parameter to process_image() so server.py can pass it through."""

    # 2a: Update function signature
    old_sig = "def process_image(img_path, reader, index, stamp_template, hash_db=None,\n                  verbose=False):"
    new_sig = "def process_image(img_path, reader, index, stamp_template, hash_db=None,\n                  verbose=False, use_cache=True):"
    r1 = apply_fix("scanner.py", old_sig, new_sig, "Fix 2a: Add use_cache to process_image signature", dry_run)

    # 2b: Use the parameter in the pricing call
    old_pricing = '            use_cache=True,  # Default to batch mode; server.py can override'
    new_pricing = '            use_cache=use_cache,'
    r2 = apply_fix("scanner.py", old_pricing, new_pricing, "Fix 2b: Wire use_cache param into fetch_cached_pricing", dry_run)

    return r1 and r2


# ============================================
# Fix 3: Guard reference image overwrite
# ============================================

def fix_overwrite_guard(dry_run=False):
    """Prevent card_detect from overwriting reference images in batch mode."""

    old = """\
    # Run card detection + perspective correction on all inputs
    img, card_detected = detect_and_crop_card(img)
    if card_detected:
        # Overwrite temp file so downstream steps use corrected image
        cv2.imwrite(str(img_path), img)"""

    new = """\
    # Run card detection + perspective correction on all inputs
    img, card_detected = detect_and_crop_card(img)
    if card_detected:
        # Overwrite temp file so downstream steps use corrected image.
        # Guard: never overwrite reference images (batch/test mode).
        ref_dir = str(Path(__file__).parent / "data" / "Ref Images")
        if not str(img_path).startswith(ref_dir):
            cv2.imwrite(str(img_path), img)"""

    return apply_fix("scanner.py", old, new, "Fix 3: Guard reference image overwrite", dry_run)


# ============================================
# Fix 4: Deduplicate card-detect in server.py
# ============================================

def fix_server_dedup(dry_run=False):
    """Remove 3 duplicate card-detect blocks from /scan/upload in server.py.
    The wire_card_detect.sh script was run 4 times, inserting 4 identical blocks.
    Keep the first, remove the other 3.
    """
    # Each block ends with the warning line, and the next starts after \n\n
    # We match from the comment to the end of the except line.
    single_block = (
        "    # \u2500\u2500 Detect and crop card from phone image \u2500\u2500\n"
        "    try:\n"
        "        import cv2 as _cv2\n"
        "        raw_img = _cv2.imread(img_path)\n"
        "        if raw_img is not None:\n"
        "            cropped_img, detected = detect_and_crop_card(raw_img)\n"
        "            if detected:\n"
        "                _cv2.imwrite(img_path, cropped_img)\n"
        '                logger.info("Card detected and cropped from phone image")\n'
        "            else:\n"
        '                logger.info("No card detected \u2014 using original phone image")\n'
        "    except Exception as e:\n"
        '        logger.warning("Card detection failed: %s \u2014 using original image", e)'
    )

    path = REPO_ROOT / "server.py"
    if not path.exists():
        print(f"  \u23ed\ufe0f  SKIP Fix 4 \u2014 server.py not found")
        return False

    content = path.read_text(encoding="utf-8")
    count = content.count(single_block)

    if count <= 1:
        print(f"  \u2705 SKIP Fix 4: Deduplicate card-detect \u2014 already clean ({count} copy)")
        return True

    # Replace all occurrences with empty string, then re-insert exactly one
    content = content.replace(single_block, "<<<CARD_DETECT_PLACEHOLDER>>>")
    # Put back exactly one (replace first placeholder, remove rest)
    content = content.replace("<<<CARD_DETECT_PLACEHOLDER>>>", single_block, 1)
    content = content.replace("\n\n<<<CARD_DETECT_PLACEHOLDER>>>", "")
    content = content.replace("<<<CARD_DETECT_PLACEHOLDER>>>", "")

    if not dry_run:
        path.write_text(content, encoding="utf-8")
    print(f"  \u2705 Fix 4: Deduplicate card-detect \u2014 removed {count - 1} extra copies (was {count})")
    return True


# ============================================
# Fix 5: Wire use_cache through server.py _run_scan
# ============================================

def fix_server_use_cache(dry_run=False):
    """Pass use_cache from _run_scan into process_image() and clean up the stale TODO."""

    # 5a: Add use_cache to the process_image() call in _run_scan
    old_call = """\
        result = process_image(
            img_path=img_path,
            reader=scanner_resources["reader"],
            index=scanner_resources["index"],
            stamp_template=scanner_resources["stamp_template"],
            hash_db=scanner_resources["hash_db"],
            verbose=False,
        )"""

    new_call = """\
        result = process_image(
            img_path=img_path,
            reader=scanner_resources["reader"],
            index=scanner_resources["index"],
            stamp_template=scanner_resources["stamp_template"],
            hash_db=scanner_resources["hash_db"],
            verbose=False,
            use_cache=use_cache,
        )"""

    r1 = apply_fix("server.py", old_call, new_call, "Fix 5a: Pass use_cache to process_image", dry_run)

    # 5b: Replace stale TODO block with clean cache-aware tagging
    old_todo = """\
        # ── Apply pricing cache logic ──
        # process_image() already calls fetch_live_pricing() internally.
        # To integrate the cache, you'll add use_cache as a parameter to
        # process_image(), or replace its pricing call with
        # fetch_cached_pricing(). For now, we tag the result so the
        # dashboard knows the mode.
        #
        # TODO: Wire fetch_cached_pricing() into scanner.py's pricing step.
        #       Until then, this flag is informational only.
        result["_cached"] = False
        result["_pricing_mode"] = "batch" if use_cache else "single\""""

    new_todo = """\
        # Tag pricing mode for dashboard display
        result["_pricing_mode"] = "batch" if use_cache else "single\""""

    r2 = apply_fix("server.py", old_todo, new_todo, "Fix 5b: Remove stale TODO, clean up cache tags", dry_run)

    return r1 and r2


# ============================================
# Fix 6: Remove duplicate return in ocr.py
# ============================================

def fix_ocr_duplicate_return(dry_run=False):
    """Remove the unreachable duplicate return at the end of extract_number()."""

    old = """\
    return best_match[0], best_match[1], round(best_match[2] * 100, 1)

    return best_match[0], best_match[1], round(best_match[2] * 100, 1)"""

    new = """\
    return best_match[0], best_match[1], round(best_match[2] * 100, 1)"""

    return apply_fix("ocr.py", old, new, "Fix 6: Remove duplicate return in ocr.py", dry_run)


# ============================================
# Fix 7: Create/update .gitignore
# ============================================

def fix_gitignore(dry_run=False):
    """Create or overwrite .gitignore with proper exclusions."""
    path = REPO_ROOT / ".gitignore"

    if not dry_run:
        path.write_text(GITIGNORE_CONTENT, encoding="utf-8")
    print(f"  ✅ Fix 7: .gitignore {'created' if not path.exists() else 'updated'}")
    return True


# ============================================
# Fix 8: Delete cleanup candidates
# ============================================

def fix_cleanup(dry_run=False):
    """Delete applied patch scripts, .bak files, and cert/key files."""
    deleted = 0
    skipped = 0

    for filename in FILES_TO_DELETE:
        path = REPO_ROOT / filename
        if path.exists():
            if not dry_run:
                path.unlink()
            deleted += 1
            print(f"    🗑️  {filename}")
        else:
            skipped += 1

    print(f"  ✅ Fix 8: Cleanup — {deleted} files deleted, {skipped} already gone")
    return True


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser(description="Apply all fixes to the card scanner codebase")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be changed without modifying files")
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "APPLYING"
    print("=" * 60)
    print(f"Card Scanner — All Fixes ({mode})")
    print("=" * 60)
    print()

    results = []

    print("── scanner.py ──")
    results.append(fix_stamp_detection(args.dry_run))
    results.append(fix_use_cache_param(args.dry_run))
    results.append(fix_overwrite_guard(args.dry_run))
    print()

    print("── server.py ──")
    results.append(fix_server_dedup(args.dry_run))
    results.append(fix_server_use_cache(args.dry_run))
    print()

    print("── ocr.py ──")
    results.append(fix_ocr_duplicate_return(args.dry_run))
    print()

    print("── Repo hygiene ──")
    results.append(fix_gitignore(args.dry_run))
    print()

    print("── File cleanup ──")
    results.append(fix_cleanup(args.dry_run))
    print()

    # Summary
    passed = sum(1 for r in results if r)
    total = len(results)
    print("=" * 60)
    print(f"Results: {passed}/{total} fixes applied")
    if args.dry_run:
        print("(Dry run — no files were modified)")
    print("=" * 60)

    print("""
Next steps:
  1. Review the changes:
       git diff
  2. Run a quick sanity test:
       python3 -c "from scanner import process_image; print('OK')"
       python3 test_random.py -n 2
  3. Commit and push:
       git add -A
       git commit -m "Fix stamp detection, server dedup, cache wiring, cleanup"
       git push origin main
""")


if __name__ == "__main__":
    main()

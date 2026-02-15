#!/usr/bin/env python3
"""
apply_ocr_fixes_v2.py — Fixes gaps from initial apply script.

Run from ~/card-scanner after apply_ocr_improvements.py.

Fixes:
  1. Missing NUMBER_EARLY_EXIT_CONF in config.py
  2. Missing VINTAGE_SET_IDS in config.py
  3. Regex double-escape fix in ocr.py
  4. pricing_cache import fallback in scanner.py
  5. card_detect.py — detect actual filename and apply min width
  6. by_name index builder in database.py (if missing)
"""

import os
import sys
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
if not (PROJECT_DIR / "server.py").exists():
    PROJECT_DIR = Path.home() / "card-scanner"
    if not (PROJECT_DIR / "server.py").exists():
        print("ERROR: Run from ~/card-scanner")
        sys.exit(1)

os.chdir(PROJECT_DIR)
print(f"Working directory: {PROJECT_DIR}\n")


def read_file(filename):
    path = PROJECT_DIR / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def write_file(filename, content):
    path = PROJECT_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ {filename} updated")


# ═══════════════════════════════════════════════════════════════
# Fix 1: config.py — Add missing constants
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("Fix 1: config.py — Missing constants")
print("=" * 60)

config = read_file("config.py")
if config:
    additions = ""

    if "NUMBER_EARLY_EXIT_CONF" not in config:
        additions += "\n# Early exit threshold for number OCR (skip remaining variants)\n"
        additions += "NUMBER_EARLY_EXIT_CONF = 0.85\n"
        print("  Adding NUMBER_EARLY_EXIT_CONF = 0.85")

    if "VINTAGE_SET_IDS" not in config:
        additions += """
# Vintage/WOTC set IDs (higher collector value, prioritized in name-only lookup)
VINTAGE_SET_IDS = {
    "base1", "base2", "base3", "base4", "base5", "base6",
    "gym1", "gym2",
    "neo1", "neo2", "neo3", "neo4",
    "si1",  # Southern Islands
    "ecard1", "ecard2", "ecard3",
    "basep",  # Wizards Promo
    "bp",     # Best of Game promo
}
"""
        print("  Adding VINTAGE_SET_IDS (WOTC-era sets)")

    if additions:
        write_file("config.py", config + additions)
    else:
        print("  Already has both constants")


# ═══════════════════════════════════════════════════════════════
# Fix 2: ocr.py — Regex escape fix
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Fix 2: ocr.py — Regex patterns")
print("=" * 60)

ocr = read_file("ocr.py")
if ocr:
    # The double backslash issue: \\\\] should be just \\]
    # In a raw string r'...' we want the literal characters \ and ]
    # which means r'[|1lI)\]7\\]' — one escaped ] and one escaped \

    old_pattern = r"re.compile(r'(\d{1,3})\s*[|1lI)\]7\\\\]\s*(\d{1,3})')"
    new_pattern = r"re.compile(r'(\d{1,3})\s*[|1lI)\]7\\]\s*(\d{1,3})')"

    if old_pattern in ocr:
        ocr = ocr.replace(old_pattern, new_pattern)
        write_file("ocr.py", ocr)
        print("  Fixed double-backslash in misread regex")
    else:
        # Try to find any malformed version
        # Check if the corrected version is already there
        if new_pattern in ocr:
            print("  Regex already correct")
        else:
            # Look for what's actually there
            matches = re.findall(r"re\.compile\(r'.*?1lI.*?'\)", ocr)
            if matches:
                print(f"  Current pattern: {matches[0]}")
                print("  Please verify manually — couldn't auto-fix")
            else:
                print("  Misread regex pattern not found (may need manual check)")
else:
    print("  ocr.py not found")


# ═══════════════════════════════════════════════════════════════
# Fix 3: scanner.py — pricing_cache import fallback
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Fix 3: scanner.py — pricing import fallback")
print("=" * 60)

scanner = read_file("scanner.py")
if scanner:
    # Replace bare import with try/except fallback
    old_import = "        from pricing_cache import fetch_cached_pricing"
    new_import = """        try:
            from pricing_cache import fetch_cached_pricing
        except ImportError:
            from pricing_justtcg import fetch_live_pricing as fetch_cached_pricing"""

    if old_import in scanner and "except ImportError" not in scanner:
        scanner = scanner.replace(old_import, new_import)
        write_file("scanner.py", scanner)
        print("  Added import fallback: pricing_cache → pricing_justtcg")
    elif "except ImportError" in scanner:
        print("  Fallback already in place")
    else:
        print("  pricing_cache import pattern not found — check manually")
else:
    print("  scanner.py not found")


# ═══════════════════════════════════════════════════════════════
# Fix 4: card_detect — Find actual filename and apply min width
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Fix 4: Card detection — Locate and patch")
print("=" * 60)

# Search for the actual card detection file
candidates = [
    "card_detect.py", "card_detector.py", "detect.py",
    "detection.py", "card_detection.py", "detect_card.py",
]

detect_file = None
for name in candidates:
    if (PROJECT_DIR / name).exists():
        detect_file = name
        break

# Also search for files containing card detection logic
if not detect_file:
    for pyfile in PROJECT_DIR.glob("*.py"):
        content = pyfile.read_text(encoding="utf-8", errors="ignore")
        if "perspectiveTransform" in content or "warpPerspective" in content:
            detect_file = pyfile.name
            print(f"  Found card detection logic in: {detect_file}")
            break

if detect_file:
    detect = read_file(detect_file)

    # Look for a minimum width check pattern
    width_patterns = [
        (r"card_w < (\d+)", "numeric width check"),
        (r"MIN.*WIDTH.*=.*(\d+)", "width constant"),
        (r"if.*width.*<.*(\d+)", "width guard"),
    ]

    found_width = False
    for pattern, desc in width_patterns:
        match = re.search(pattern, detect, re.IGNORECASE)
        if match:
            old_val = match.group(1)
            print(f"  Found {desc}: {old_val}px")
            found_width = True
            if int(old_val) < 480:
                # Replace with CARD_MIN_OUTPUT_WIDTH from config
                if "CARD_MIN_OUTPUT_WIDTH" not in detect:
                    # Add import
                    if "from config import" in detect:
                        detect = detect.replace(
                            "from config import",
                            "from config import CARD_MIN_OUTPUT_WIDTH,"
                        )
                    else:
                        detect = "from config import CARD_MIN_OUTPUT_WIDTH\n" + detect

                # Replace the hardcoded value
                full_match = match.group(0)
                new_match = full_match.replace(old_val, "CARD_MIN_OUTPUT_WIDTH")
                detect = detect.replace(full_match, new_match, 1)
                write_file(detect_file, detect)
                print(f"  Replaced {old_val} → CARD_MIN_OUTPUT_WIDTH (480)")
            else:
                print(f"  Width already ≥ 480, no change needed")
            break

    if not found_width:
        print(f"  No minimum width check found in {detect_file}")
        print("  You may want to add a resize step manually after perspective correction")
else:
    print("  No card detection file found. Available .py files:")
    for f in sorted(PROJECT_DIR.glob("*.py")):
        print(f"    {f.name}")
    print("  → If card detection is in a different file, let me know")


# ═══════════════════════════════════════════════════════════════
# Fix 5: database.py — Ensure by_name index exists
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Fix 5: database.py — by_name index check")
print("=" * 60)

db = read_file("database.py")
if db:
    if "by_name" in db:
        print("  'by_name' already referenced in database.py")
        # Check if it's being built in load_database or build_index
        if "by_name" in db and ("index[\"by_name\"]" in db or "index['by_name']" in db or "'by_name'" in db):
            print("  Index appears to be constructed — verify it includes all cards")
        else:
            print("  WARNING: 'by_name' referenced but may not be built in index")
            print("  → lookup_by_name() needs index['by_name'] = {name_lower: [card_dicts]}")
    else:
        print("  WARNING: 'by_name' not found in database.py")
        print("  → Need to add by_name index construction to load_database()")
        print("  → Expected structure: index['by_name'] = {'boldore': [{card1}, {card2}], ...}")
        print("  → Let me know if you want me to generate that patch")
else:
    print("  database.py not found")


# ═══════════════════════════════════════════════════════════════
# Fix 6: Verify fetch_cached_pricing signature compatibility
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Fix 6: Pricing function signature check")
print("=" * 60)

pricing_file = None
for name in ["pricing_cache.py", "pricing_justtcg.py"]:
    if (PROJECT_DIR / name).exists():
        pricing_file = name
        break

if pricing_file:
    pricing = read_file(pricing_file)
    # Check if function accepts the kwargs we're passing
    if "def fetch_cached_pricing" in pricing or "def fetch_live_pricing" in pricing:
        func_match = re.search(r"def (fetch_cached_pricing|fetch_live_pricing)\((.*?)\)", pricing, re.DOTALL)
        if func_match:
            params = func_match.group(2)
            print(f"  Found {func_match.group(1)}({params[:80]}...)")
            if "use_cache" not in params and "**kwargs" not in params:
                print("  WARNING: function doesn't accept 'use_cache' param")
                print("  → Scanner passes use_cache=True — may cause TypeError")
                print("  → Either add **kwargs to the function or remove use_cache from scanner.py call")
            if "is_1st_edition" not in params and "**kwargs" not in params:
                print("  WARNING: function doesn't accept 'is_1st_edition' param")
            if "rarity" not in params and "**kwargs" not in params:
                print("  WARNING: function doesn't accept 'rarity' param")
    else:
        print(f"  No fetch function found in {pricing_file}")
else:
    print("  Neither pricing_cache.py nor pricing_justtcg.py found")
    print("  Available .py files:")
    for f in sorted(PROJECT_DIR.glob("*.py")):
        print(f"    {f.name}")


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("V2 FIXES COMPLETE")
print("=" * 60)
print("""
Review with:  git diff
Revert all:   git checkout -- .

If by_name index needs building, let me know and I'll generate that patch.
If pricing function signatures don't match, share the function header and I'll fix.
""")

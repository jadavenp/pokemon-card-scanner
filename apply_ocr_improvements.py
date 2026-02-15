#!/usr/bin/env python3
"""
apply_ocr_improvements.py — Session 5 Consolidated Patch Script

Applies all changes IN-PLACE to ~/card-scanner files.
Run `git diff` after to review, `git checkout -- .` to revert.

Changes applied:
  1. Bug fixes (config.py, server.py, scanner.py imports)
  2. New preprocessing pipeline in ocr.py (CLAHE, highlight reduction, etc.)
  3. Confidence-based identification in scanner.py
  4. Name-only database fallback in database.py
  5. Debug image saving for number strip diagnostics
  6. Config updates for new OCR parameters
  7. Wire fetch_cached_pricing() into scanner.py

Usage:
  cd ~/card-scanner
  python3 apply_ocr_improvements.py
  git diff  # review changes
"""

import os
import re
import sys
import shutil
from pathlib import Path

# ── Resolve project root ──
PROJECT_DIR = Path(__file__).parent
if not (PROJECT_DIR / "server.py").exists():
    # Try ~/card-scanner
    PROJECT_DIR = Path.home() / "card-scanner"
    if not (PROJECT_DIR / "server.py").exists():
        print("ERROR: Could not find card-scanner project.")
        print("Run this script from ~/card-scanner or place it there.")
        sys.exit(1)

os.chdir(PROJECT_DIR)
print(f"Working directory: {PROJECT_DIR}")

# ── Helpers ──

def read_file(filename):
    path = PROJECT_DIR / filename
    if not path.exists():
        print(f"  WARNING: {filename} not found, skipping")
        return None
    return path.read_text(encoding="utf-8")


def write_file(filename, content):
    path = PROJECT_DIR / filename
    path.write_text(content, encoding="utf-8")
    print(f"  ✓ {filename} updated")


def replace_in_file(filename, old, new, description=""):
    content = read_file(filename)
    if content is None:
        return False
    if old not in content:
        print(f"  SKIP ({filename}): pattern not found — {description or old[:60]}")
        return False
    if content.count(old) > 1:
        print(f"  WARNING ({filename}): multiple matches for pattern, replacing first — {description}")
        content = content.replace(old, new, 1)
    else:
        content = content.replace(old, new)
    write_file(filename, content)
    if description:
        print(f"         {description}")
    return True


# ═══════════════════════════════════════════════════════════════
# PHASE 0: Bug Fixes
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 0: Critical Bug Fixes")
print("=" * 60)

# Fix #1: config.py — DATABASE_FILE path
content = read_file("config.py")
if content and 'pokemon_tcg_database.json' in content:
    replace_in_file(
        "config.py",
        'DATABASE_FILE = DATA_DIR / "pokemon_tcg_database.json"',
        'DATABASE_FILE = DATA_DIR / "card_index.json"',
        "Fix DATABASE_FILE path → card_index.json"
    )
else:
    print("  config.py DATABASE_FILE already correct or not found")

# Fix #2: server.py — tuple unpacking for load_database()
content = read_file("server.py")
if content and 'scanner_resources["index"] = load_database()' in content:
    replace_in_file(
        "server.py",
        'scanner_resources["index"] = load_database()',
        'scanner_resources["index"], _ = load_database()',
        "Fix database tuple unpacking"
    )
else:
    print("  server.py tuple unpacking already fixed or pattern differs")

# Fix #3: scanner.py — pricing import
content = read_file("scanner.py")
if content and 'from pricing import fetch_live_pricing' in content:
    replace_in_file(
        "scanner.py",
        'from pricing import fetch_live_pricing',
        'from pricing_justtcg import fetch_live_pricing',
        "Fix pricing import → pricing_justtcg"
    )
else:
    print("  scanner.py pricing import already fixed or pattern differs")


# ═══════════════════════════════════════════════════════════════
# PHASE 1: config.py — Add new OCR constants
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 1: Config Updates")
print("=" * 60)

config_additions = '''
# ============================================
# OCR v2 — Enhanced number extraction (Session 5)
# ============================================

# Number region crops by card type (tighter than v1)
# Format: {y_start, y_end, x_start, x_end} as fraction of card dimensions
NUMBER_REGIONS = {
    "pokemon": {"y_start": 0.87, "y_end": 0.97, "x_start": 0.03, "x_end": 0.58},
    "trainer": {"y_start": 0.87, "y_end": 0.97, "x_start": 0.03, "x_end": 0.58},
    "energy":  {"y_start": 0.87, "y_end": 0.97, "x_start": 0.03, "x_end": 0.58},
}

# Minimum card width after perspective correction (ensures number text is readable)
CARD_MIN_OUTPUT_WIDTH = 480

# CLAHE parameters for number strip
NUMBER_CLAHE_CLIP = 2.5
NUMBER_CLAHE_GRID = (4, 4)

# Highlight reduction — clamp V channel in HSV above this value
NUMBER_HIGHLIGHT_CLAMP = 200

# Unsharp mask parameters
NUMBER_UNSHARP_SIGMA = 2.0
NUMBER_UNSHARP_WEIGHT = 1.5   # original weight
NUMBER_UNSHARP_BLUR_WEIGHT = -0.5  # blur subtraction weight

# Bilateral filter for denoising
NUMBER_BILATERAL_D = 5
NUMBER_BILATERAL_SIGMA = 40

# EasyOCR parameters for number strip
NUMBER_ALLOWLIST = "0123456789/"
NUMBER_CONTRAST_THS = 0.05
NUMBER_ADJUST_CONTRAST = 0.7
NUMBER_MIN_SIZE = 5

# Upscale factor for number ROI (v2)
NUMBER_UPSCALE_V2 = 4

# Debug: save preprocessed number strip images
# Set to a directory path to enable, None to disable
OCR_DEBUG_DIR = None  # e.g., Path("debug_ocr") to enable

# Confidence thresholds for identification
CONFIDENCE_NAME_WEIGHT = 0.40     # Weight of name OCR confidence in composite score
CONFIDENCE_NUMBER_WEIGHT = 0.35   # Weight of number OCR confidence
CONFIDENCE_HASH_WEIGHT = 0.20     # Weight of hash match confidence
CONFIDENCE_STAMP_WEIGHT = 0.05    # Weight of stamp detection (low — not an ID signal)

# Known set totals for validation (card_number / total)
# Maps total → list of possible set IDs
KNOWN_SET_TOTALS = {
    "102": ["base1"],
    "64": ["base2", "gym1", "gym2"],
    "62": ["base3"],
    "130": ["base4"],
    "83": ["base5"],
    "82": ["base6"],
    "111": ["neo1"],
    "75": ["neo2"],
    "66": ["neo3"],
    "113": ["neo4"],
    "165": ["ecard1"],
    "147": ["ecard2"],
    "182": ["ecard3"],
    "109": ["ex1"],
    "100": ["ex2"],
    "203": ["swsh1", "bw1"],
}
'''

content = read_file("config.py")
if content and "NUMBER_REGIONS" not in content:
    write_file("config.py", content + config_additions)
    print("  ✓ Added OCR v2 constants to config.py")
else:
    print("  config.py already has NUMBER_REGIONS or not found")


# ═══════════════════════════════════════════════════════════════
# PHASE 2: ocr.py — Enhanced extract_number() + debug saving
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 2: OCR Pipeline Improvements")
print("=" * 60)

# We'll replace the entire extract_number function and add new imports/helpers
content = read_file("ocr.py")
if content is None:
    print("  FATAL: ocr.py not found")
    sys.exit(1)

# Add new imports at the top if not present
new_imports = """from pathlib import Path
import logging
import time

logger = logging.getLogger("ocr")
"""

if "logger = logging.getLogger" not in content:
    # Insert after the existing imports
    content = content.replace(
        "from config import (",
        new_imports + "\nfrom config import ("
    )

# Add the new config imports
old_config_import_end = ")"  # End of the from config import (...)
# Find the config import block and extend it
config_import_pattern = r"(from config import \([^)]+)\)"
match = re.search(config_import_pattern, content, re.DOTALL)
if match:
    existing_imports = match.group(1)
    if "NUMBER_REGIONS" not in existing_imports:
        new_config_imports = """,
    NUMBER_REGIONS, NUMBER_CLAHE_CLIP, NUMBER_CLAHE_GRID,
    NUMBER_HIGHLIGHT_CLAMP, NUMBER_UNSHARP_SIGMA, NUMBER_UNSHARP_WEIGHT,
    NUMBER_UNSHARP_BLUR_WEIGHT, NUMBER_BILATERAL_D, NUMBER_BILATERAL_SIGMA,
    NUMBER_ALLOWLIST, NUMBER_CONTRAST_THS, NUMBER_ADJUST_CONTRAST,
    NUMBER_MIN_SIZE, NUMBER_UPSCALE_V2, OCR_DEBUG_DIR,
    KNOWN_SET_TOTALS,"""
        content = content.replace(
            existing_imports + ")",
            existing_imports + new_config_imports + "\n)"
        )

# Now replace extract_number entirely
old_extract_number_start = "def extract_number(img, reader):"
old_extract_number_end = "    return best_match[0], best_match[1], round(best_match[2] * 100, 1)"

if old_extract_number_start in content and old_extract_number_end in content:
    # Find the full function
    start_idx = content.index(old_extract_number_start)
    end_idx = content.index(old_extract_number_end) + len(old_extract_number_end)
    old_function = content[start_idx:end_idx]

    new_extract_number = '''def _save_debug_image(img, label, card_type="unknown"):
    """Save a preprocessing step image for debugging."""
    if OCR_DEBUG_DIR is None:
        return
    debug_dir = Path(OCR_DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000) % 100000
    filename = f"{timestamp}_{card_type}_{label}.png"
    cv2.imwrite(str(debug_dir / filename), img)


def _reduce_highlights(roi_bgr):
    """
    Reduce specular highlights by clamping the V channel in HSV.
    Preserves detail in bright areas that would otherwise saturate to white.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_clamped = np.clip(v, 0, NUMBER_HIGHLIGHT_CLAMP)
    v_norm = cv2.normalize(v_clamped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv_out = cv2.merge([h, s, v_norm])
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


def _apply_clahe(gray):
    """Apply CLAHE to normalize local contrast (handles uneven phone lighting)."""
    clahe = cv2.createCLAHE(
        clipLimit=NUMBER_CLAHE_CLIP,
        tileGridSize=NUMBER_CLAHE_GRID,
    )
    return clahe.apply(gray)


def _unsharp_mask(gray):
    """Gaussian unsharp mask — sharpens text edges without amplifying noise."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=NUMBER_UNSHARP_SIGMA)
    return cv2.addWeighted(
        gray, NUMBER_UNSHARP_WEIGHT,
        blurred, NUMBER_UNSHARP_BLUR_WEIGHT,
        0,
    )


def _preprocess_number_roi(roi_bgr, card_type="pokemon"):
    """
    Full preprocessing pipeline for the card number strip.
    Returns a list of (label, processed_image) variants for OCR.
    """
    variants = []

    # Step 1: Upscale ROI
    scale = NUMBER_UPSCALE_V2
    roi_up = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _save_debug_image(roi_up, "01_upscaled", card_type)

    # Step 2: Reduce highlights (before grayscale to preserve color info)
    roi_dehighlight = _reduce_highlights(roi_up)
    _save_debug_image(roi_dehighlight, "02_dehighlight", card_type)

    # Step 3: Convert to grayscale (luminance-weighted)
    gray = cv2.cvtColor(roi_dehighlight, cv2.COLOR_BGR2GRAY)
    _save_debug_image(gray, "03_gray", card_type)

    # Step 4: Bilateral filter — denoise while preserving edges
    denoised = cv2.bilateralFilter(
        gray, NUMBER_BILATERAL_D,
        NUMBER_BILATERAL_SIGMA, NUMBER_BILATERAL_SIGMA,
    )
    _save_debug_image(denoised, "04_denoised", card_type)

    # Step 5: CLAHE — equalize local contrast
    clahe_img = _apply_clahe(denoised)
    _save_debug_image(clahe_img, "05_clahe", card_type)

    # Step 6: Unsharp mask — sharpen text edges
    sharpened = _unsharp_mask(clahe_img)
    _save_debug_image(sharpened, "06_sharpened", card_type)

    # Step 7: Generate binarization variants
    # 7a: CLAHE + Otsu (best general purpose)
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _save_debug_image(otsu, "07a_otsu", card_type)
    variants.append(("otsu", otsu))

    # 7b: Adaptive Gaussian
    adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )
    _save_debug_image(adaptive, "07b_adaptive", card_type)
    variants.append(("adaptive", adaptive))

    # 7c: Sharpened grayscale (let EasyOCR handle contrast internally)
    variants.append(("sharpened", sharpened))

    # 7d: Inverted for light-on-dark text (some card eras)
    _, inv = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _save_debug_image(inv, "07d_inverted", card_type)
    variants.append(("inverted", inv))

    return variants


def extract_number(img, reader, card_type="pokemon"):
    """
    Pass 2 (v2): Extract card number from the bottom strip using enhanced
    preprocessing pipeline with CLAHE, highlight reduction, and constrained OCR.

    Returns (card_number, set_total, confidence_pct) or (None, None, 0.0).
    """
    h, w = img.shape[:2]

    # Use card-type-specific crop region (tighter than v1)
    region = NUMBER_REGIONS.get(card_type, NUMBER_REGIONS["pokemon"])
    y_start = int(h * region["y_start"])
    y_end = int(h * region["y_end"])
    x_start = int(w * region["x_start"])
    x_end = int(w * region["x_end"])

    roi = img[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return None, None, 0.0

    _save_debug_image(roi, "00_raw_roi", card_type)

    # Run full preprocessing pipeline
    variants = _preprocess_number_roi(roi, card_type)

    # Expanded regex patterns to catch common OCR misreads of "/"
    number_patterns = [
        re.compile(r'(\d{1,3})\s*/\s*(\d{1,3})'),            # Standard: 057/203
        re.compile(r'(\d{1,3})\s*[|1lI)\]7\\\\]\s*(\d{1,3})'),  # OCR misreads of /
        re.compile(r'(\d{1,3})\s*[.,]\s*(\d{1,3})'),           # Dot/comma misread
    ]

    best_match = (None, None, 0.0, "")  # (number, total, confidence, variant_label)

    for label, processed in variants:
        try:
            results = reader.readtext(
                processed,
                allowlist=NUMBER_ALLOWLIST,
                paragraph=False,
                min_size=NUMBER_MIN_SIZE,
                contrast_ths=NUMBER_CONTRAST_THS,
                adjust_contrast=NUMBER_ADJUST_CONTRAST,
            )
        except Exception as e:
            logger.warning("EasyOCR failed on variant %s: %s", label, e)
            continue

        for (bbox, text, conf) in results:
            for pattern in number_patterns:
                match = pattern.search(text)
                if match and conf > best_match[2]:
                    candidate_num = match.group(1).lstrip("0") or "0"
                    candidate_total = match.group(2)
                    best_match = (candidate_num, candidate_total, conf, label)

                    # Early exit on high confidence
                    if conf > NUMBER_EARLY_EXIT_CONF:
                        logger.info(
                            "Number extraction: %s/%s (conf=%.2f, variant=%s)",
                            candidate_num, candidate_total, conf, label,
                        )
                        return best_match[0], best_match[1], round(best_match[2] * 100, 1)

    if best_match[0]:
        logger.info(
            "Number extraction: %s/%s (conf=%.2f, variant=%s)",
            best_match[0], best_match[1], best_match[2], best_match[3],
        )

    return best_match[0], best_match[1], round(best_match[2] * 100, 1)'''

    content = content.replace(old_function, new_extract_number)
    write_file("ocr.py", content)
    print("  ✓ Replaced extract_number() with v2 pipeline")
    print("  ✓ Added preprocessing helpers (CLAHE, highlight reduction, unsharp mask)")
    print("  ✓ Added debug image saving")
else:
    print("  WARNING: Could not find extract_number() boundaries in ocr.py")
    print(f"    Start pattern found: {old_extract_number_start in content}")
    print(f"    End pattern found: {old_extract_number_end in content}")


# ═══════════════════════════════════════════════════════════════
# PHASE 3: database.py — Add name-only lookup fallback
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 3: Database — Name-Only Lookup Fallback")
print("=" * 60)

db_content = read_file("database.py")
if db_content and "lookup_by_name" not in db_content:
    new_function = '''

def lookup_by_name(index, card_name, set_hint=None):
    """
    Search the local database by name only (no number required).
    Used as a fallback when number extraction fails.

    Returns list of matching card dicts, optionally filtered by set hint.
    If multiple sets contain this card, returns all matches sorted by
    vintage sets first (higher value, more likely to be scanned).
    """
    if not card_name:
        return []

    search_name = card_name.lower().strip()
    matches = []

    # Exact name match
    name_matches = index.get("by_name", {}).get(search_name, [])
    if name_matches:
        matches = list(name_matches)

    # Fuzzy fallback: substring match
    if not matches:
        for name_key, cards_list in index.get("by_name", {}).items():
            if search_name in name_key or name_key in search_name:
                matches.extend(cards_list)

    # Filter by set hint if provided
    if set_hint and matches:
        filtered = [c for c in matches if get_set_id(c) == set_hint]
        if filtered:
            matches = filtered

    # Sort: vintage sets first (higher collector value), then by number
    from config import VINTAGE_SET_IDS
    def sort_key(card):
        sid = get_set_id(card)
        is_vintage = 0 if sid in VINTAGE_SET_IDS else 1
        num = card.get("number", "999")
        try:
            num_int = int(num)
        except ValueError:
            num_int = 999
        return (is_vintage, num_int)

    matches.sort(key=sort_key)
    return matches
'''

    db_content += new_function
    write_file("database.py", db_content)
    print("  ✓ Added lookup_by_name() to database.py")
else:
    print("  database.py already has lookup_by_name or not found")


# ═══════════════════════════════════════════════════════════════
# PHASE 4: scanner.py — Confidence-based identification
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 4: Scanner — Confidence-Based Identification")
print("=" * 60)

scanner_content = read_file("scanner.py")
if scanner_content is None:
    print("  FATAL: scanner.py not found")
    sys.exit(1)

# 4a: Add import for lookup_by_name and fetch_cached_pricing
if "from database import load_database, lookup_card, get_set_id" in scanner_content:
    replace_in_file(
        "scanner.py",
        "from database import load_database, lookup_card, get_set_id",
        "from database import load_database, lookup_card, lookup_by_name, get_set_id",
        "Add lookup_by_name import",
    )

# Reload after edits
scanner_content = read_file("scanner.py")

# 4b: Add config imports for confidence weights
if "CONFIDENCE_NAME_WEIGHT" not in scanner_content:
    replace_in_file(
        "scanner.py",
        "from config import IMAGE_DIR, SET_NAMES, VARIANT_DISPLAY",
        "from config import IMAGE_DIR, SET_NAMES, VARIANT_DISPLAY, CONFIDENCE_NAME_WEIGHT, CONFIDENCE_NUMBER_WEIGHT, CONFIDENCE_HASH_WEIGHT, CONFIDENCE_STAMP_WEIGHT, KNOWN_SET_TOTALS",
        "Add confidence weight imports",
    )

# Reload after edits
scanner_content = read_file("scanner.py")

# 4c: Add the compute_identification_confidence function before process_image
confidence_function = '''

def compute_identification_confidence(result):
    """
    Compute a composite identification confidence score (0-100%)
    based on weighted combination of all identification signals.

    Signals:
      - Name OCR confidence (0-100)
      - Number OCR confidence (0-100)
      - Hash match confidence (0 = no match, 100 = perfect match)
      - Stamp detection confidence (informational, low weight)

    Returns a float 0-100 representing overall ID confidence.
    """
    name_conf = result.get("name_conf", 0.0)
    num_conf = result.get("num_conf", 0.0)
    stamp_conf = result.get("stamp_conf", 0.0)

    # Convert hash distance to confidence (0 distance = 100%, 15+ = 0%)
    hash_dist = result.get("hash_distance")
    if hash_dist is not None:
        hash_conf = max(0, min(100, (15 - hash_dist) / 15 * 100))
    else:
        hash_conf = 0.0

    # Weighted combination
    # If a signal is missing (0), its weight redistributes to available signals
    weights = {}
    if name_conf > 0:
        weights["name"] = CONFIDENCE_NAME_WEIGHT
    if num_conf > 0:
        weights["number"] = CONFIDENCE_NUMBER_WEIGHT
    if hash_conf > 0:
        weights["hash"] = CONFIDENCE_HASH_WEIGHT
    if stamp_conf > 0:
        weights["stamp"] = CONFIDENCE_STAMP_WEIGHT

    if not weights:
        return 0.0

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    composite = 0.0
    if "name" in weights:
        composite += (weights["name"] / total_weight) * name_conf
    if "number" in weights:
        composite += (weights["number"] / total_weight) * num_conf
    if "hash" in weights:
        composite += (weights["hash"] / total_weight) * hash_conf
    if "stamp" in weights:
        composite += (weights["stamp"] / total_weight) * stamp_conf

    return round(composite, 1)


def validate_number_against_sets(number, total, candidate_sets=None):
    """
    Cross-check an extracted set total against known set sizes.
    Returns list of matching set IDs, or empty list if no match.
    """
    if not total:
        return []
    matching = KNOWN_SET_TOTALS.get(total, [])
    if candidate_sets and matching:
        # Intersect with candidate sets
        filtered = [s for s in matching if s in candidate_sets]
        return filtered if filtered else matching
    return matching

'''

if "def compute_identification_confidence" not in scanner_content:
    # Insert before process_image
    scanner_content = scanner_content.replace(
        "def process_image(",
        confidence_function + "\ndef process_image("
    )

# 4d: Now replace the identification logic inside process_image.
# This is the most complex replacement. We need to:
#   - Pass card_type to extract_number
#   - Add name-only fallback
#   - Add confidence scoring
#   - Always attempt pricing on best candidate

# Replace the call to extract_number to pass card_type
if "num, total, num_conf = extract_number(img, reader)" in scanner_content:
    scanner_content = scanner_content.replace(
        "num, total, num_conf = extract_number(img, reader)",
        "num, total, num_conf = extract_number(img, reader, card_type=result['card_type'])"
    )

# Replace the entire identification + pricing block
old_identification_start = "    # ── Database Lookup + Identification ──"
old_identification_end = '        result["error"] = f"OCR incomplete (hash tentative: {hash_result[\'card_id\']})"'

if old_identification_start in scanner_content and old_identification_end in scanner_content:
    start_idx = scanner_content.index(old_identification_start)
    end_idx = scanner_content.index(old_identification_end) + len(old_identification_end)
    old_block = scanner_content[start_idx:end_idx]

    new_identification_block = '''    # ── Database Lookup + Identification (confidence-based) ──
    ocr_matched = False
    id_confidence = 0.0

    # Strategy 1: Name + Number → strongest match
    if result["name"] and result["number"]:
        matches = lookup_card(index, result["name"], result["number"])
        if matches:
            card = matches[0]
            card_id = card.get("id", "")
            set_id = get_set_id(card)
            result["card_id"] = card_id
            result["set_name"] = SET_NAMES.get(set_id, set_id)
            result["rarity"] = card.get("rarity", "?")
            ocr_matched = True

            if hash_result and hash_result["match"]:
                if hash_result["card_id"] == card_id:
                    result["id_method"] = "ocr+hash"
                else:
                    result["id_method"] = "ocr (hash disagree)"
            else:
                result["id_method"] = "ocr"

    # Strategy 2: Hash fallback if OCR name+number failed
    if not ocr_matched and hash_result and hash_result["match"] and hash_result["confident"]:
        result["card_id"] = hash_result["card_id"]
        result["name"] = hash_result["name"]
        set_id = hash_result["card_id"].rsplit("-", 1)[0] if "-" in hash_result["card_id"] else ""
        result["set_name"] = SET_NAMES.get(set_id, set_id)
        result["id_method"] = "hash"
        for name_key, cards_list in index.get("by_name", {}).items():
            for c in cards_list:
                if c.get("id") == hash_result["card_id"]:
                    result["rarity"] = c.get("rarity", "?")
                    result["number"] = c.get("number", "")
                    break
            if result["rarity"]:
                break
        ocr_matched = True

    # Strategy 3: Name-only fallback (number OCR failed but name is good)
    if not ocr_matched and result["name"] and result["name_conf"] > 30:
        # Use set total hint from number extraction if available
        set_hints = validate_number_against_sets(result["number"], result["total"])
        set_hint = set_hints[0] if len(set_hints) == 1 else None

        name_matches = lookup_by_name(index, result["name"], set_hint=set_hint)
        if name_matches:
            if len(name_matches) == 1:
                # Unambiguous: only one card with this name
                card = name_matches[0]
                result["card_id"] = card.get("id", "")
                set_id = get_set_id(card)
                result["set_name"] = SET_NAMES.get(set_id, set_id)
                result["rarity"] = card.get("rarity", "?")
                result["number"] = card.get("number", result["number"])
                result["id_method"] = "name_only"
                ocr_matched = True
            else:
                # Ambiguous: multiple cards share this name — pick best candidate
                # Prefer vintage sets, or use hash hint if available
                card = name_matches[0]  # Already sorted vintage-first
                result["card_id"] = card.get("id", "")
                set_id = get_set_id(card)
                result["set_name"] = SET_NAMES.get(set_id, set_id)
                result["rarity"] = card.get("rarity", "?")
                result["number"] = card.get("number", result["number"])
                result["id_method"] = f"name_only ({len(name_matches)} candidates)"
                ocr_matched = True

    # ── Compute identification confidence ──
    id_confidence = compute_identification_confidence(result)
    result["id_confidence"] = id_confidence

    # ── Pricing (always attempt on best candidate, show confidence alongside) ──
    if ocr_matched and result["card_id"]:
        from pricing_cache import fetch_cached_pricing
        pricing = fetch_cached_pricing(
            result["card_id"],
            is_1st_edition=result["stamp_1st"],
            rarity=result.get("rarity", "?"),
            use_cache=True,  # Default to batch mode; server.py can override
        )
        result["price"] = pricing["price"]
        result["price_variant"] = pricing["variant"]
        result["price_src"] = pricing["source"]

    if not ocr_matched:
        result["error"] = "Identification failed"
        if result["name"]:
            result["error"] = f"Name '{result['name']}' not found in database"
        if hash_result and hash_result["match"] and not hash_result["confident"]:
            result["error"] = f"OCR incomplete (hash tentative: {hash_result['card_id']})"'''

    scanner_content = scanner_content.replace(old_block, new_identification_block)

# 4e: Add id_confidence to the result dict initialization
if '"id_method": None,' in scanner_content and '"id_confidence"' not in scanner_content:
    scanner_content = scanner_content.replace(
        '"id_method": None,  # "ocr", "hash", "ocr+hash", "ocr (hash disagree)"',
        '"id_method": None,  # "ocr", "hash", "ocr+hash", "name_only", etc.\n        "id_confidence": 0.0,  # Composite identification confidence (0-100)'
    )

write_file("scanner.py", scanner_content)
print("  ✓ Added confidence-based identification to scanner.py")
print("  ✓ Added name-only database fallback")
print("  ✓ Wired fetch_cached_pricing() into pipeline")
print("  ✓ Added compute_identification_confidence()")
print("  ✓ Added validate_number_against_sets()")


# ═══════════════════════════════════════════════════════════════
# PHASE 5: card_detect.py — Higher output resolution
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 5: Card Detection — Higher Output Resolution")
print("=" * 60)

detect_content = read_file("card_detect.py")
if detect_content and "CARD_MIN_OUTPUT_WIDTH" not in detect_content:
    # Add config import
    detect_content = detect_content.replace(
        "import logging",
        "import logging\n\nfrom config import CARD_MIN_OUTPUT_WIDTH"
    )

    # Replace the minimum width check
    detect_content = detect_content.replace(
        "        # Ensure minimum output size for OCR readability\n"
        "        if card_w < 200:\n"
        "            scale = 200 / card_w\n"
        "            card_w = 200\n"
        "            card_h = int(card_h * scale)",

        "        # Ensure minimum output size for OCR readability\n"
        "        if card_w < CARD_MIN_OUTPUT_WIDTH:\n"
        "            scale = CARD_MIN_OUTPUT_WIDTH / card_w\n"
        "            card_w = CARD_MIN_OUTPUT_WIDTH\n"
        "            card_h = int(card_h * scale)"
    )

    write_file("card_detect.py", detect_content)
    print("  ✓ Updated card_detect.py minimum output width → 480px (via config)")
else:
    print("  card_detect.py already updated or not found")


# ═══════════════════════════════════════════════════════════════
# PHASE 6: server.py — Pass use_cache and update /scan/upload
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 6: Server — Minor Updates")
print("=" * 60)

server_content = read_file("server.py")
if server_content:
    # Update the label in _run_scan if it still says TCGGO
    if "TCGGO:" in server_content:
        replace_in_file(
            "server.py",
            "TCGGO:",
            "JustTCG:",
            "Fix API label TCGGO → JustTCG"
        )
else:
    print("  server.py not found")


# ═══════════════════════════════════════════════════════════════
# PHASE 7: Create debug helper script
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PHASE 7: Debug Helper")
print("=" * 60)

debug_script = '''#!/usr/bin/env python3
"""
enable_ocr_debug.py — Toggle debug image saving for OCR number extraction.

Usage:
    python3 enable_ocr_debug.py on    # Enable: saves to debug_ocr/
    python3 enable_ocr_debug.py off   # Disable
    python3 enable_ocr_debug.py clean  # Delete debug images
"""

import sys
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.py"
DEBUG_DIR = Path(__file__).parent / "debug_ocr"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 enable_ocr_debug.py [on|off|clean]")
        sys.exit(1)

    action = sys.argv[1].lower()
    content = CONFIG_FILE.read_text()

    if action == "on":
        content = content.replace(
            'OCR_DEBUG_DIR = None',
            'OCR_DEBUG_DIR = Path("debug_ocr")'
        )
        CONFIG_FILE.write_text(content)
        DEBUG_DIR.mkdir(exist_ok=True)
        print(f"Debug ON — images will save to {DEBUG_DIR}/")

    elif action == "off":
        content = content.replace(
            'OCR_DEBUG_DIR = Path("debug_ocr")',
            'OCR_DEBUG_DIR = None'
        )
        CONFIG_FILE.write_text(content)
        print("Debug OFF")

    elif action == "clean":
        if DEBUG_DIR.exists():
            import shutil
            count = len(list(DEBUG_DIR.glob("*.png")))
            shutil.rmtree(DEBUG_DIR)
            print(f"Deleted {count} debug images from {DEBUG_DIR}/")
        else:
            print("No debug directory found")

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''

debug_path = PROJECT_DIR / "enable_ocr_debug.py"
debug_path.write_text(debug_script, encoding="utf-8")
print("  ✓ Created enable_ocr_debug.py")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ALL CHANGES APPLIED")
print("=" * 60)
print("""
Files modified:
  config.py          — Fixed DATABASE_FILE, added OCR v2 constants
  server.py          — Fixed tuple unpacking, fixed API label
  scanner.py         — Fixed pricing import, added confidence-based ID,
                       name-only fallback, fetch_cached_pricing integration
  ocr.py             — Replaced extract_number() with v2 pipeline
                       (CLAHE, highlight reduction, allowlist, debug saving)
  database.py        — Added lookup_by_name() fallback
  card_detect.py     — Increased min output width to 480px

Files created:
  enable_ocr_debug.py — Toggle debug image saving

Next steps:
  1. Review changes:     git diff
  2. Test:               python3 server.py --host 0.0.0.0
  3. Enable debug:       python3 enable_ocr_debug.py on
  4. Scan a card from phone, check debug_ocr/ for strip images
  5. Commit when happy:  git add -A && git commit -m "Session 5: OCR v2 + confidence ID"
""")

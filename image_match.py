"""
image_match.py
Perceptual image hashing for art-based card identification.

Used as a second identification pathway alongside OCR:
  - If OCR succeeds: image hash confirms the match
  - If OCR fails: image hash provides primary identification
  - If both disagree: flag for manual review

Hash types used:
  - pHash (perceptual): DCT-based, best for general similarity
  - dHash (difference): gradient-based, fast, good for near-duplicates
  - wHash (wavelet): wavelet-based, robust to minor edits

Hamming distance thresholds:
  - 0:     identical
  - 1-8:   very likely match (same card, different scan quality)
  - 9-15:  possible match (similar artwork, check manually)
  - 16+:   different card
"""

import json

import cv2
import imagehash
import numpy as np
from PIL import Image

from config import DATA_DIR

HASH_DB_FILE = DATA_DIR / "hash_database.json"

# Hamming distance thresholds
DISTANCE_CONFIDENT = 8    # Below this = confident match
DISTANCE_POSSIBLE = 15    # Below this = possible match, above = no match

# Weights for combining hash distances into a single score
# pHash is most reliable for card art; dHash is fastest; wHash adds robustness
HASH_WEIGHTS = {"phash": 0.5, "dhash": 0.3, "whash": 0.2}

# ============================================
# Era-specific art region crop profiles
# Must match build_hash_db.py exactly
# ============================================
ART_CROPS = {
    "wotc": {"y_start": 0.12, "y_end": 0.49, "x_start": 0.10, "x_end": 0.90},
    "ecard": {"y_start": 0.12, "y_end": 0.52, "x_start": 0.08, "x_end": 0.92},
    "ex": {"y_start": 0.12, "y_end": 0.52, "x_start": 0.08, "x_end": 0.92},
    "dp_hgss": {"y_start": 0.12, "y_end": 0.50, "x_start": 0.10, "x_end": 0.90},
    "bw_xy": {"y_start": 0.11, "y_end": 0.52, "x_start": 0.08, "x_end": 0.92},
    "sm_swsh": {"y_start": 0.10, "y_end": 0.52, "x_start": 0.07, "x_end": 0.93},
    "sv": {"y_start": 0.10, "y_end": 0.52, "x_start": 0.07, "x_end": 0.93},
    "full_art": {"y_start": 0.03, "y_end": 0.85, "x_start": 0.05, "x_end": 0.95},
}

# Map set ID prefixes to crop eras
ERA_MAP = {
    "base": "wotc", "gym": "wotc", "neo": "wotc", "si": "wotc", "basep": "wotc",
    "ecard": "ecard",
    "ex": "ex",
    "dp": "dp_hgss", "pl": "dp_hgss", "hgss": "dp_hgss", "dpp": "dp_hgss", "hsp": "dp_hgss",
    "bw": "bw_xy", "xy": "bw_xy", "bwp": "bw_xy", "xyp": "bw_xy", "g": "bw_xy", "dc": "bw_xy",
    "sm": "sm_swsh", "swsh": "sm_swsh", "smp": "sm_swsh", "swshp": "sm_swsh",
    "sma": "sm_swsh", "det": "sm_swsh",
    "sv": "sv", "svp": "sv",
    "mcd": "sm_swsh", "pop": "ex", "col": "dp_hgss",
    "cel": "sm_swsh", "me": "sv",
}


def _get_era(set_id):
    """Determine the era for a set ID based on prefix matching."""
    if set_id in ERA_MAP:
        return ERA_MAP[set_id]
    for prefix in sorted(ERA_MAP.keys(), key=len, reverse=True):
        if set_id.startswith(prefix):
            return ERA_MAP[prefix]
    return "sm_swsh"


def load_hash_database():
    """
    Load the hash database from disk.
    Returns dict: {card_id: {name, set_id, era, phash, dhash, whash}} or None.
    """
    if not HASH_DB_FILE.exists():
        return None

    try:
        with open(HASH_DB_FILE, "r") as f:
            data = json.load(f)
        hashes = data.get("hashes", {})
        if not hashes:
            return None

        # Convert hex strings to imagehash objects for fast comparison
        db = {}
        for card_id, entry in hashes.items():
            db[card_id] = {
                "name": entry["name"],
                "set_id": entry["set_id"],
                "era": entry.get("era", _get_era(entry["set_id"])),
                "phash": imagehash.hex_to_hash(entry["phash"]),
                "dhash": imagehash.hex_to_hash(entry["dhash"]),
                "whash": imagehash.hex_to_hash(entry["whash"]),
            }
        return db
    except Exception:
        return None


def crop_art_from_scan(img_bgr, era="sm_swsh"):
    """
    Crop the art region from a scanned card image (OpenCV BGR format).

    Args:
        img_bgr: numpy array (OpenCV BGR format)
        era: crop profile era key (default: sm_swsh)

    Returns:
        PIL Image of the cropped art region.
    """
    h, w = img_bgr.shape[:2]
    crop = ART_CROPS.get(era, ART_CROPS["sm_swsh"])

    left = int(w * crop["x_start"])
    right = int(w * crop["x_end"])
    top = int(h * crop["y_start"])
    bottom = int(h * crop["y_end"])

    art_rgb = cv2.cvtColor(img_bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    return Image.fromarray(art_rgb)


def compute_scan_hashes(art_img):
    """
    Compute perceptual hashes for a scanned art region.
    Returns dict with phash, dhash, whash as imagehash objects.
    """
    return {
        "phash": imagehash.phash(art_img),
        "dhash": imagehash.dhash(art_img),
        "whash": imagehash.whash(art_img),
    }


def find_match(scan_hashes_by_era, hash_db, top_n=3):
    """
    Find the closest matching card(s) in the hash database.
    Compares scan hashes against each ref card using the ref card's era crop.

    Args:
        scan_hashes_by_era: dict of era -> {phash, dhash, whash}
        hash_db: loaded hash database from load_hash_database()
        top_n: number of top candidates to return

    Returns list of dicts, sorted by distance (best first).
    Returns empty list if no match below DISTANCE_POSSIBLE.
    """
    if not hash_db:
        return []

    candidates = []

    for card_id, ref in hash_db.items():
        ref_era = ref.get("era", "sm_swsh")
        scan_hashes = scan_hashes_by_era.get(ref_era)
        if scan_hashes is None:
            # Fallback: use the default era hash
            scan_hashes = scan_hashes_by_era.get("sm_swsh", list(scan_hashes_by_era.values())[0])

        distances = {
            "phash": scan_hashes["phash"] - ref["phash"],
            "dhash": scan_hashes["dhash"] - ref["dhash"],
            "whash": scan_hashes["whash"] - ref["whash"],
        }

        weighted = sum(
            distances[h] * HASH_WEIGHTS[h] for h in HASH_WEIGHTS
        )

        if weighted <= DISTANCE_POSSIBLE:
            candidates.append({
                "card_id": card_id,
                "name": ref["name"],
                "set_id": ref["set_id"],
                "distance": round(weighted, 1),
                "distances": distances,
                "confident": weighted <= DISTANCE_CONFIDENT,
            })

    candidates.sort(key=lambda x: x["distance"])
    return candidates[:top_n]


def match_card_image(img_bgr, hash_db):
    """
    High-level interface: given a scanned card image, find the best match.

    Computes hashes for each unique era crop profile so that distance
    comparisons are apples-to-apples with how the hash DB was built.

    Args:
        img_bgr: numpy array (OpenCV BGR format) — full card image
        hash_db: loaded hash database

    Returns dict:
        {
            "match": True/False,
            "card_id": "base1-58" or None,
            "name": "Pikachu" or None,
            "distance": 3.2 or None,
            "confident": True/False,
            "candidates": [top 3 matches],
        }
    """
    if hash_db is None:
        return {
            "match": False, "card_id": None, "name": None,
            "distance": None, "confident": False, "candidates": [],
        }

    # Find which eras exist in the hash database
    eras_in_db = set()
    for ref in hash_db.values():
        eras_in_db.add(ref.get("era", "sm_swsh"))

    # Compute scan hashes for each era's crop profile
    scan_hashes_by_era = {}
    for era in eras_in_db:
        art = crop_art_from_scan(img_bgr, era=era)
        scan_hashes_by_era[era] = compute_scan_hashes(art)

    candidates = find_match(scan_hashes_by_era, hash_db)

    if candidates and candidates[0]["confident"]:
        best = candidates[0]
        return {
            "match": True,
            "card_id": best["card_id"],
            "name": best["name"],
            "distance": best["distance"],
            "confident": True,
            "candidates": candidates,
        }
    elif candidates:
        best = candidates[0]
        return {
            "match": True,
            "card_id": best["card_id"],
            "name": best["name"],
            "distance": best["distance"],
            "confident": False,
            "candidates": candidates,
        }
    else:
        return {
            "match": False, "card_id": None, "name": None,
            "distance": None, "confident": False, "candidates": [],
        }

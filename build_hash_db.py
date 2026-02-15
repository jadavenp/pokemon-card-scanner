"""
build_hash_db.py
Build a perceptual hash database from LOCAL reference card images.

Reads images from data/Ref Images/ (already downloaded).
Uses card_index.json to know which cards exist and have images.
Computes perceptual hashes (pHash, dHash, wHash) on the art region.
Saves to data/hash_database.json.

Usage:
    cd ~/card-scanner && source venv/bin/activate
    python3 build_hash_db.py                    # Base Set only (default)
    python3 build_hash_db.py --sets base1,base2 # specific sets
    python3 build_hash_db.py --all              # all sets with images
    python3 build_hash_db.py --all --rebuild    # force rebuild from scratch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import imagehash
except ImportError:
    print("Missing dependency: pip install imagehash")
    sys.exit(1)

# ============================================
# Configuration
# ============================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CARD_INDEX_FILE = DATA_DIR / "card_index.json"
HASH_DB_FILE = DATA_DIR / "hash_database.json"
REF_IMAGES_DIR = DATA_DIR / "Ref Images"

# ============================================
# Era-specific art region crop profiles
# Percentages of image dimensions (y_start, y_end, x_start, x_end)
# ============================================
ART_CROPS = {
    # WOTC era: Base Set through Neo Destiny + Southern Islands
    "wotc": {
        "y_start": 0.12, "y_end": 0.49,
        "x_start": 0.10, "x_end": 0.90,
    },
    # e-Card era: Expedition, Aquapolis, Skyridge
    "ecard": {
        "y_start": 0.12, "y_end": 0.52,
        "x_start": 0.08, "x_end": 0.92,
    },
    # EX era: Ruby & Sapphire through Power Keepers
    "ex": {
        "y_start": 0.12, "y_end": 0.52,
        "x_start": 0.08, "x_end": 0.92,
    },
    # DP/Platinum/HGSS era
    "dp_hgss": {
        "y_start": 0.12, "y_end": 0.50,
        "x_start": 0.10, "x_end": 0.90,
    },
    # BW/XY era
    "bw_xy": {
        "y_start": 0.11, "y_end": 0.52,
        "x_start": 0.08, "x_end": 0.92,
    },
    # SM/SWSH era
    "sm_swsh": {
        "y_start": 0.10, "y_end": 0.52,
        "x_start": 0.07, "x_end": 0.93,
    },
    # SV era (Scarlet & Violet)
    "sv": {
        "y_start": 0.10, "y_end": 0.52,
        "x_start": 0.07, "x_end": 0.93,
    },
    # Full-art / alt-art / VMAX / VSTAR — hash full card minus border
    "full_art": {
        "y_start": 0.03, "y_end": 0.85,
        "x_start": 0.05, "x_end": 0.95,
    },
}

# Map set ID prefixes to crop eras
ERA_MAP = {
    # WOTC
    "base": "wotc", "gym": "wotc", "neo": "wotc", "si": "wotc",
    "basep": "wotc",
    # e-Card
    "ecard": "ecard",
    # EX
    "ex": "ex",
    # DP / Platinum / HGSS
    "dp": "dp_hgss", "pl": "dp_hgss", "hgss": "dp_hgss",
    "dpp": "dp_hgss", "hsp": "dp_hgss",
    # BW / XY
    "bw": "bw_xy", "xy": "bw_xy", "bwp": "bw_xy", "xyp": "bw_xy",
    "g": "bw_xy", "dc": "bw_xy",
    # SM / SWSH
    "sm": "sm_swsh", "swsh": "sm_swsh", "smp": "sm_swsh", "swshp": "sm_swsh",
    "sma": "sm_swsh", "det": "sm_swsh",
    # SV
    "sv": "sv", "svp": "sv",
    # Promos / McDonalds / misc — default to modern
    "mcd": "sm_swsh", "pop": "ex", "col": "dp_hgss",
    "cel": "sm_swsh", "me": "sv",
}


def get_era(set_id):
    """Determine the era for a set ID based on prefix matching."""
    # Try exact match first
    if set_id in ERA_MAP:
        return ERA_MAP[set_id]

    # Try progressively shorter prefixes
    # Sort by longest prefix first for best match
    for prefix in sorted(ERA_MAP.keys(), key=len, reverse=True):
        if set_id.startswith(prefix):
            return ERA_MAP[prefix]

    # Default fallback
    return "sm_swsh"


def get_art_crop(set_id):
    """Return the crop profile for a given set."""
    era = get_era(set_id)
    return ART_CROPS.get(era, ART_CROPS["sm_swsh"])


def crop_art_region(img, crop_profile):
    """Crop the art region from a card image."""
    w, h = img.size
    left = int(w * crop_profile["x_start"])
    right = int(w * crop_profile["x_end"])
    top = int(h * crop_profile["y_start"])
    bottom = int(h * crop_profile["y_end"])
    return img.crop((left, top, right, bottom))


def compute_hashes(img):
    """Compute perceptual hashes for an image."""
    return {
        "phash": str(imagehash.phash(img)),
        "dhash": str(imagehash.dhash(img)),
        "whash": str(imagehash.whash(img)),
    }


def load_card_index():
    """Load card_index.json and return cards grouped by set."""
    if not CARD_INDEX_FILE.exists():
        print(f"Error: {CARD_INDEX_FILE} not found.")
        print("Run build_card_index.py first.")
        sys.exit(1)

    with open(CARD_INDEX_FILE, "r", encoding="utf-8") as f:
        card_index = json.load(f)

    return card_index


def get_cards_for_sets(card_index, target_sets):
    """Get cards that have images for the target sets."""
    by_id = card_index["index"]["by_id"]
    cards = []

    for card_id, card in by_id.items():
        set_id = card.get("set", {}).get("id", "")
        if target_sets and set_id not in target_sets:
            continue
        if not card.get("has_image", False):
            continue
        cards.append({
            "id": card_id,
            "name": card.get("name", ""),
            "set_id": set_id,
            "number": card.get("number", ""),
        })

    return sorted(cards, key=lambda x: x["id"])


def get_all_set_ids(card_index):
    """Get all set IDs that have at least one card with an image."""
    sets = set()
    for card in card_index["cards"]:
        if card.get("has_image", False):
            sets.add(card["set"]["id"])
    return sets


def build_hash_database(card_index, target_sets, rebuild=False, verbose=False):
    """Build the hash database from local reference images."""
    cards = get_cards_for_sets(card_index, target_sets)
    print(f"  {len(cards)} cards to process ({len(target_sets)} sets)")
    print()

    # Load existing or start fresh
    hash_db = {
        "version": 2,
        "built_at": "",
        "card_count": 0,
        "hashes": {},
    }

    if not rebuild and HASH_DB_FILE.exists():
        try:
            with open(HASH_DB_FILE, "r") as f:
                existing = json.load(f)
            hash_db["hashes"] = existing.get("hashes", {})
            print(f"  Resuming: {len(hash_db['hashes'])} cards already hashed")
        except Exception:
            pass
    elif rebuild:
        print("  Rebuilding from scratch (--rebuild)")

    processed = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for i, card in enumerate(cards):
        card_id = card["id"]
        set_id = card["set_id"]
        name = card["name"]

        # Skip if already hashed (unless rebuilding)
        if not rebuild and card_id in hash_db["hashes"]:
            skipped += 1
            continue

        # Load local image
        img_path = REF_IMAGES_DIR / f"{card_id}.png"
        if not img_path.exists():
            failed += 1
            if verbose:
                print(f"  [{i+1}/{len(cards)}] MISSING {card_id}")
            continue

        try:
            img = Image.open(img_path)
        except Exception as e:
            failed += 1
            if verbose:
                print(f"  [{i+1}/{len(cards)}] ERROR {card_id}: {e}")
            continue

        # Crop art region and compute hashes
        crop_profile = get_art_crop(set_id)
        art = crop_art_region(img, crop_profile)
        hashes = compute_hashes(art)

        hash_db["hashes"][card_id] = {
            "name": name,
            "set_id": set_id,
            "era": get_era(set_id),
            "phash": hashes["phash"],
            "dhash": hashes["dhash"],
            "whash": hashes["whash"],
        }

        processed += 1

        if verbose:
            print(f"  [{i+1}/{len(cards)}] {card_id} ({name}) — OK")

        # Progress every 500 cards (non-verbose mode)
        if not verbose and processed % 500 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining_cards = len(cards) - i - 1 - skipped
            remaining_time = remaining_cards / rate if rate > 0 else 0
            print(f"  {processed} hashed, {skipped} skipped, "
                  f"{i+1}/{len(cards)} processed "
                  f"({rate:.0f}/sec, ~{remaining_time:.0f}s remaining)")

        # Save checkpoint every 2000 cards
        if processed % 2000 == 0:
            hash_db["card_count"] = len(hash_db["hashes"])
            hash_db["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(HASH_DB_FILE, "w") as f:
                json.dump(hash_db, f)

    # Final save
    hash_db["card_count"] = len(hash_db["hashes"])
    hash_db["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(HASH_DB_FILE, "w") as f:
        json.dump(hash_db, f)

    elapsed = time.time() - t_start
    file_size = os.path.getsize(HASH_DB_FILE) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"HASH DATABASE BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed:   {processed}")
    print(f"  Skipped:     {skipped} (already hashed)")
    print(f"  Failed:      {failed}")
    print(f"  Total:       {hash_db['card_count']} hashes in database")
    print(f"  Time:        {elapsed:.1f}s ({processed/elapsed:.0f} cards/sec)" if elapsed > 0 else "")
    print(f"  File size:   {file_size:.1f} MB")
    print(f"  Saved to:    {HASH_DB_FILE}")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description="Build perceptual hash database from local reference card images"
    )
    parser.add_argument(
        "--sets", type=str, default="base1",
        help="Comma-separated set IDs (default: base1)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Process all sets that have images"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force rebuild from scratch (ignore existing hashes)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-card progress"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Pokemon TCG Hash Database Builder")
    print("=" * 60)
    print()

    # Load card index
    print("Loading card index...")
    card_index = load_card_index()
    total_cards = card_index["meta"]["card_count"]
    with_images = card_index["meta"].get("cards_with_images", "?")
    print(f"  {total_cards} cards, {with_images} with images")

    # Determine target sets
    if args.all:
        target_sets = get_all_set_ids(card_index)
        print(f"\nProcessing ALL {len(target_sets)} sets with images...")
    else:
        target_sets = set(s.strip() for s in args.sets.split(","))
        print(f"\nProcessing sets: {', '.join(sorted(target_sets))}")

    # Build
    print()
    build_hash_database(card_index, target_sets, rebuild=args.rebuild, verbose=args.verbose)


if __name__ == "__main__":
    main()

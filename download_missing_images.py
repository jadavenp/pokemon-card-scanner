"""
download_missing_images.py
Downloads reference images that exist in card_index.json but are missing from data/Ref Images/.

Source: images.pokemontcg.io CDN (no auth, no rate limits)
Format: https://images.pokemontcg.io/{set_id}/{number}.png

Usage:
    python3 download_missing_images.py              # Download all missing
    python3 download_missing_images.py --dry-run     # Show what would be downloaded
    python3 download_missing_images.py --set swshp   # Only download missing from one set
    python3 download_missing_images.py --limit 100   # Stop after 100 downloads
"""

import json
import os
import sys
import time
import argparse
import requests
from collections import defaultdict

# ============================================
# CONFIG
# ============================================
DATA_DIR = "data"
REF_IMAGES_DIR = os.path.join(DATA_DIR, "Ref Images")
CARD_INDEX_FILE = os.path.join(DATA_DIR, "card_index.json")
CDN_BASE = "https://images.pokemontcg.io"

# Download settings
TIMEOUT = 15
RETRY_COUNT = 2
RETRY_DELAY = 2
# Small delay between requests to be polite to the CDN
REQUEST_DELAY = 0.1


def load_card_index():
    """Load card_index.json and return the by_id index."""
    if not os.path.exists(CARD_INDEX_FILE):
        print(f"Error: {CARD_INDEX_FILE} not found.")
        print("Run build_card_index.py first.")
        sys.exit(1)

    print(f"Loading {CARD_INDEX_FILE}...")
    with open(CARD_INDEX_FILE, "r", encoding="utf-8") as f:
        card_index = json.load(f)

    by_id = card_index.get("index", {}).get("by_id", {})
    print(f"  {len(by_id)} cards in index")
    return by_id


def scan_existing_images():
    """Get set of card IDs that already have images on disk."""
    if not os.path.exists(REF_IMAGES_DIR):
        print(f"Warning: {REF_IMAGES_DIR} not found. Will create it.")
        os.makedirs(REF_IMAGES_DIR, exist_ok=True)
        return set()

    existing = set()
    for fname in os.listdir(REF_IMAGES_DIR):
        if fname.lower().endswith(".png"):
            # Strip .png to get card ID (e.g., "base1-1.png" -> "base1-1")
            card_id = fname[:-4]
            existing.add(card_id)

    print(f"  {len(existing)} images already on disk")
    return existing


def find_missing(by_id, existing, set_filter=None):
    """Find card IDs in the index that don't have images."""
    missing = []
    for card_id in sorted(by_id.keys()):
        if card_id not in existing:
            if set_filter:
                # Card ID format: {set_id}-{number}
                card_set = by_id[card_id].get("set", {}).get("id", "")
                if card_set != set_filter:
                    continue
            missing.append(card_id)

    return missing


def get_image_url(card_id, by_id):
    """Get the image URL from the card index data.

    Uses the 'small' image URL from the card's images field.
    Falls back to constructing from card ID if no URL stored.
    """
    card = by_id.get(card_id, {})
    images = card.get("images", {})

    # Prefer the small PNG from the index (confirmed working URLs)
    if images.get("small"):
        return images["small"]

    # Fallback: construct from card ID
    parts = card_id.rsplit("-", 1)
    if len(parts) != 2:
        return None
    set_id, number = parts
    return f"{CDN_BASE}/{set_id}/{number}.png"


def download_image(url, dest_path, retries=RETRY_COUNT):
    """Download a single image with retries."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                return True, resp.status_code
            elif resp.status_code == 404:
                return False, 404
            else:
                if attempt < retries:
                    time.sleep(RETRY_DELAY)
                    continue
                return False, resp.status_code
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(RETRY_DELAY)
                continue
            return False, str(e)

    return False, "max retries"


def download_missing(missing, by_id, dry_run=False, limit=None):
    """Download all missing images."""
    total = len(missing)
    if limit:
        missing = missing[:limit]
        print(f"\nLimited to {limit} of {total} missing images")
    else:
        print(f"\n{total} images to download")

    if dry_run:
        print("\n--- DRY RUN (no downloads) ---\n")
        by_set = defaultdict(list)
        for card_id in missing:
            parts = card_id.rsplit("-", 1)
            set_id = parts[0] if len(parts) == 2 else "unknown"
            by_set[set_id].append(card_id)

        for set_id in sorted(by_set.keys()):
            cards = by_set[set_id]
            print(f"  {set_id:<20s}  {len(cards):>4d} to download")
        print(f"\n  Total: {len(missing)} images")
        return

    os.makedirs(REF_IMAGES_DIR, exist_ok=True)

    downloaded = 0
    failed_404 = 0
    failed_other = 0
    start_time = time.time()

    for i, card_id in enumerate(missing):
        url = get_image_url(card_id, by_id)
        if not url:
            print(f"  [{i+1}/{len(missing)}] SKIP {card_id} — can't build URL")
            failed_other += 1
            continue

        dest = os.path.join(REF_IMAGES_DIR, f"{card_id}.png")

        success, status = download_image(url, dest)

        if success:
            downloaded += 1
            if downloaded % 50 == 0 or i == len(missing) - 1:
                elapsed = time.time() - start_time
                rate = downloaded / elapsed if elapsed > 0 else 0
                remaining = (len(missing) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(missing)}] Downloaded {downloaded} "
                      f"({rate:.1f}/sec, ~{remaining:.0f}s remaining)")
        elif status == 404:
            failed_404 += 1
            if failed_404 <= 10:
                print(f"  [{i+1}/{len(missing)}] 404 {card_id}")
        else:
            failed_other += 1
            if failed_other <= 10:
                print(f"  [{i+1}/{len(missing)}] FAIL {card_id} — {status}")

        time.sleep(REQUEST_DELAY)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Downloaded:    {downloaded}")
    print(f"  404 (missing): {failed_404}")
    print(f"  Other errors:  {failed_other}")
    print(f"  Time:          {elapsed:.0f}s")
    if downloaded > 0:
        print(f"  Rate:          {downloaded / elapsed:.1f} images/sec")

    if failed_404 > 0:
        print(f"\n  Note: {failed_404} cards returned 404. These may be promos or special")
        print(f"  cards not hosted on the standard CDN path.")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Download missing Pokemon TCG reference images")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--set", type=str, default=None, help="Only download missing images for a specific set ID")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to download")
    args = parser.parse_args()

    print("=" * 60)
    print("Pokemon TCG — Download Missing Reference Images")
    print("=" * 60)
    print()

    # Load index
    by_id = load_card_index()

    # Scan existing
    print(f"\nScanning {REF_IMAGES_DIR}...")
    existing = scan_existing_images()

    # Find missing
    print("\nFinding missing images...")
    missing = find_missing(by_id, existing, set_filter=args.set)
    print(f"  {len(missing)} images missing")

    if not missing:
        print("\nAll images present! Nothing to download.")
        return

    # Download
    download_missing(missing, by_id, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()

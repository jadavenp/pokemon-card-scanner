"""
flag_image_status.py
Adds a "has_image" field to every card in card_index.json based on
whether a matching file exists in data/Ref Images/.

Usage:
    python3 flag_image_status.py
"""

import json
import os
import time

DATA_DIR = "data"
REF_IMAGES_DIR = os.path.join(DATA_DIR, "Ref Images")
CARD_INDEX_FILE = os.path.join(DATA_DIR, "card_index.json")

def main():
    print("=" * 60)
    print("Flag Image Status in Card Index")
    print("=" * 60)

    # Load index
    print(f"\nLoading {CARD_INDEX_FILE}...")
    with open(CARD_INDEX_FILE, "r", encoding="utf-8") as f:
        card_index = json.load(f)

    cards = card_index["cards"]
    by_id = card_index["index"]["by_id"]
    print(f"  {len(cards)} cards in index")

    # Scan images on disk
    print(f"\nScanning {REF_IMAGES_DIR}...")
    image_ids = set()
    for fname in os.listdir(REF_IMAGES_DIR):
        if fname.lower().endswith(".png"):
            image_ids.add(fname[:-4])
    print(f"  {len(image_ids)} images on disk")

    # Flag every card in the cards list
    has_count = 0
    missing_count = 0
    for card in cards:
        if card["id"] in image_ids:
            card["has_image"] = True
            has_count += 1
        else:
            card["has_image"] = False
            missing_count += 1

    # Also flag in the by_id index
    for card_id, card in by_id.items():
        card["has_image"] = card_id in image_ids

    # Update meta
    card_index["meta"]["image_flagged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    card_index["meta"]["cards_with_images"] = has_count
    card_index["meta"]["cards_without_images"] = missing_count

    # Save
    print(f"\nSaving {CARD_INDEX_FILE}...")
    with open(CARD_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(card_index, f)

    file_size = os.path.getsize(CARD_INDEX_FILE) / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"  With images:     {has_count}")
    print(f"  Without images:  {missing_count}")
    print(f"  File size:       {file_size:.1f} MB")

if __name__ == "__main__":
    main()

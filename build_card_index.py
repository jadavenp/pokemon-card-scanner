"""
build_card_index.py
Downloads the PokemonTCG GitHub data zip and builds a text-only card index.
Output: data/card_index.json

This is the TEXT-ONLY card database. It does NOT contain images.
Images live separately in data/ref_images/ and are joined by card ID
(e.g., "base1-1" matches base1-1.png).

Usage:
    python3 build_card_index.py
    python3 build_card_index.py --no-download   # Skip download, use existing zip
"""

import json
import os
import sys
import time
import zipfile
import glob
import argparse

# ============================================
# CONFIG
# ============================================
DATA_DIR = "data"
ZIP_URL = "https://github.com/PokemonTCG/pokemon-tcg-data/archive/refs/heads/master.zip"
ZIP_FILE = os.path.join(DATA_DIR, "pokemon_tcg_data.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "pokemon_tcg_extract")
OUTPUT_FILE = os.path.join(DATA_DIR, "card_index.json")


def download_zip():
    """Download the GitHub zip if not present or if forced."""
    import requests

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(ZIP_FILE):
        size_mb = os.path.getsize(ZIP_FILE) / (1024 * 1024)
        print(f"Existing zip found: {ZIP_FILE} ({size_mb:.1f} MB)")
        print("Delete it and re-run to force fresh download, or use --no-download.")
        return True

    print(f"Downloading from GitHub...")
    print(f"  URL: {ZIP_URL}")
    try:
        resp = requests.get(ZIP_URL, stream=True, timeout=60)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(ZIP_FILE, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = (downloaded / total) * 100
                    print(f"\r  Downloaded: {downloaded / (1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)

        print(f"\n  Saved to: {ZIP_FILE}")
        return True

    except Exception as e:
        print(f"  Download failed: {e}")
        if os.path.exists(ZIP_FILE):
            os.remove(ZIP_FILE)
        return False


def extract_zip():
    """Extract the zip and return the base directory path."""
    if not os.path.exists(ZIP_FILE):
        print(f"Error: {ZIP_FILE} not found.")
        print(f"Run without --no-download to fetch it, or manually download from:")
        print(f"  {ZIP_URL}")
        sys.exit(1)

    print(f"Extracting {ZIP_FILE}...")
    if os.path.exists(EXTRACT_DIR):
        import shutil
        shutil.rmtree(EXTRACT_DIR)

    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        z.extractall(EXTRACT_DIR)

    # Find extracted folder (includes branch name like "pokemon-tcg-data-master")
    extracted = glob.glob(os.path.join(EXTRACT_DIR, "pokemon-tcg-data-*"))
    if not extracted:
        print("Error: Could not find extracted data folder.")
        sys.exit(1)

    return extracted[0]


def load_sets(base_dir):
    """Load set metadata from sets/en.json."""
    sets_file = os.path.join(base_dir, "sets", "en.json")
    if not os.path.exists(sets_file):
        print(f"Warning: {sets_file} not found. Set metadata will be limited.")
        return {}

    with open(sets_file, "r", encoding="utf-8") as f:
        sets_list = json.load(f)

    sets_by_id = {}
    for s in sets_list:
        set_id = s.get("id", "")
        sets_by_id[set_id] = {
            "id": set_id,
            "name": s.get("name", ""),
            "series": s.get("series", ""),
            "printedTotal": s.get("printedTotal", 0),
            "total": s.get("total", 0),
            "releaseDate": s.get("releaseDate", ""),
            "ptcgoCode": s.get("ptcgoCode", ""),
        }

    print(f"  Loaded {len(sets_by_id)} sets from sets/en.json")
    return sets_by_id


def load_cards(base_dir, sets_by_id):
    """Load all card data from cards/en/*.json and enrich with set metadata."""
    cards_dir = os.path.join(base_dir, "cards", "en")
    if not os.path.exists(cards_dir):
        print(f"Error: {cards_dir} not found.")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(cards_dir, "*.json")))
    print(f"  Found {len(json_files)} set files in cards/en/")

    all_cards = []
    sets_seen = set()

    for filepath in json_files:
        set_id_from_file = os.path.splitext(os.path.basename(filepath))[0]
        sets_seen.add(set_id_from_file)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                cards = json.load(f)
        except json.JSONDecodeError as e:
            print(f"  Warning: Could not parse {filepath}: {e}")
            continue

        if not isinstance(cards, list):
            print(f"  Warning: {filepath} is not a list, skipping.")
            continue

        # Get set metadata from sets/en.json
        set_meta = sets_by_id.get(set_id_from_file, {})

        for card in cards:
            # Build a clean card record — text only, no image URLs
            card_id = card.get("id", "")
            number = card.get("number", "")
            name = card.get("name", "")
            supertype = card.get("supertype", "")

            # The card's own "set" field (from the card JSON)
            card_set = card.get("set", {})
            if isinstance(card_set, str):
                card_set = {"id": card_set}

            # Merge: prefer sets/en.json metadata, fall back to card-embedded set data
            resolved_set = {
                "id": set_meta.get("id", card_set.get("id", set_id_from_file)),
                "name": set_meta.get("name", card_set.get("name", "")),
                "series": set_meta.get("series", card_set.get("series", "")),
                "printedTotal": set_meta.get("printedTotal", card_set.get("printedTotal", 0)),
                "total": set_meta.get("total", card_set.get("total", 0)),
                "releaseDate": set_meta.get("releaseDate", card_set.get("releaseDate", "")),
                "ptcgoCode": set_meta.get("ptcgoCode", card_set.get("ptcgoCode", "")),
            }

            clean_card = {
                "id": card_id,
                "name": name,
                "number": number,
                "supertype": supertype,
                "subtypes": card.get("subtypes", []),
                "types": card.get("types", []),
                "hp": card.get("hp", ""),
                "rarity": card.get("rarity", ""),
                "artist": card.get("artist", ""),
                "set": resolved_set,
                "nationalPokedexNumbers": card.get("nationalPokedexNumbers", []),
                "evolvesFrom": card.get("evolvesFrom", ""),
                "evolvesTo": card.get("evolvesTo", []),
                "attacks": card.get("attacks", []),
                "weaknesses": card.get("weaknesses", []),
                "resistances": card.get("resistances", []),
                "retreatCost": card.get("retreatCost", []),
                "abilities": card.get("abilities", []),
                "rules": card.get("rules", []),
                "flavorText": card.get("flavorText", ""),
                "regulationMark": card.get("regulationMark", ""),
                "legalities": card.get("legalities", {}),
                "convertedRetreatCost": card.get("convertedRetreatCost", 0),
                "ancientTrait": card.get("ancientTrait", None),
                "images": card.get("images", {}),
            }

            # Strip None values to keep file lean
            clean_card = {k: v for k, v in clean_card.items() if v is not None}

            all_cards.append(clean_card)

    print(f"  Total cards loaded: {len(all_cards)}")
    print(f"  Sets in card files: {len(sets_seen)}")

    return all_cards


def build_indexes(all_cards):
    """Build lookup indexes for fast searching."""
    print("Building indexes...")

    by_id = {}           # "base1-1" -> card
    by_set_number = {}   # "base1/1" -> card
    by_name = {}         # "charizard" -> [cards...]
    by_number_name = {}  # "4/charizard" -> [cards...]

    for card in all_cards:
        card_id = card["id"]
        set_id = card["set"]["id"]
        number = card["number"]
        name = card["name"].lower()

        # Primary: by card ID (matches image filenames)
        by_id[card_id] = card

        # By set + number
        set_key = f"{set_id}/{number}"
        by_set_number[set_key] = card

        # By name (many cards share names across sets)
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(card_id)

        # By number + name
        num_name = f"{number}/{name}"
        if num_name not in by_number_name:
            by_number_name[num_name] = []
        by_number_name[num_name].append(card_id)

    print(f"  by_id:          {len(by_id)} entries")
    print(f"  by_set_number:  {len(by_set_number)} entries")
    print(f"  by_name:        {len(by_name)} unique names")
    print(f"  by_number_name: {len(by_number_name)} entries")

    return {
        "by_id": by_id,
        "by_set_number": by_set_number,
        "by_name": by_name,
        "by_number_name": by_number_name,
    }


def save_index(all_cards, indexes, sets_by_id):
    """Save the card index to JSON."""
    card_index = {
        "meta": {
            "description": "Pokemon TCG Card Index — text only, no images",
            "source": "https://github.com/PokemonTCG/pokemon-tcg-data",
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "card_count": len(all_cards),
            "set_count": len(sets_by_id),
            "note": "Image filenames in data/ref_images/ match card IDs (e.g., base1-1.png = card ID base1-1)",
        },
        "sets": sets_by_id,
        "cards": all_cards,
        "index": indexes,
    }

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(card_index, f)

    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")

    return card_index


def cleanup():
    """Remove extracted temp files (keep the zip for future rebuilds)."""
    import shutil
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
        print("Cleaned up temp extraction folder.")


def print_summary(card_index):
    """Print a nice summary of what was built."""
    sets = card_index["sets"]
    cards = card_index["cards"]

    print("\n" + "=" * 60)
    print("CARD INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"  Output:       {OUTPUT_FILE}")
    print(f"  Total cards:  {len(cards)}")
    print(f"  Total sets:   {len(sets)}")
    print(f"  Index keys:   by_id, by_set_number, by_name, by_number_name")
    print()

    # Show a few sets as examples
    print("Sample sets:")
    sample_sets = list(sets.values())[:5]
    for s in sample_sets:
        print(f"  {s['id']:12s}  {s['name']:35s}  ({s['printedTotal']} cards, {s['releaseDate']})")
    if len(sets) > 5:
        print(f"  ... and {len(sets) - 5} more")
    print()

    # Show a few cards as examples
    print("Sample cards:")
    sample_cards = cards[:5]
    for c in sample_cards:
        print(f"  {c['id']:20s}  {c['name']:20s}  #{c['number']:4s}  {c['set']['name']}")
    if len(cards) > 5:
        print(f"  ... and {len(cards) - 5} more")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Build Pokemon TCG card index from GitHub data")
    parser.add_argument("--no-download", action="store_true", help="Skip download, use existing zip")
    args = parser.parse_args()

    print("=" * 60)
    print("Pokemon TCG Card Index Builder")
    print("=" * 60)
    print()

    # Step 1: Download
    if not args.no_download:
        if not download_zip():
            print("Cannot proceed without data. Exiting.")
            sys.exit(1)
    else:
        print("Skipping download (--no-download)")

    # Step 2: Extract
    base_dir = extract_zip()

    # Step 3: Load sets
    print("\nLoading set metadata...")
    sets_by_id = load_sets(base_dir)

    # Step 4: Load cards
    print("\nLoading card data...")
    all_cards = load_cards(base_dir, sets_by_id)

    if not all_cards:
        print("No cards found! Check the zip file and data structure.")
        sys.exit(1)

    # Step 5: Build indexes
    indexes = build_indexes(all_cards)

    # Step 6: Save
    card_index = save_index(all_cards, indexes, sets_by_id)

    # Step 7: Cleanup
    cleanup()

    # Step 8: Summary
    print_summary(card_index)


if __name__ == "__main__":
    main()

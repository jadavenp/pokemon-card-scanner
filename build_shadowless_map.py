"""
build_shadowless_map.py
Build mapping from pokemontcg.io base1 card IDs to TCGPlayer
Base Set (Shadowless) product IDs (group 1663).

TCGPlayer stores Base Set 1st Edition pricing under the "Base Set (Shadowless)"
group, which has separate product IDs from the regular Base Set group (604).

Output: data/tcgplayer_shadowless_map.json
Format: {
    "base1-4": {"tcgplayer_id": 106999, "name": "Charizard", "confidence": "exact"},
    ...
}

Usage:
    python3 build_shadowless_map.py
"""

import json
import re
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests is required. Install with: pip install requests")
    sys.exit(1)

# ── Configuration ──

TCGCSV_BASE = "https://tcgcsv.com/tcgplayer/3"
SHADOWLESS_GROUP_ID = 1663  # "Base Set (Shadowless)"
CARD_INDEX_PATH = Path("data/card_index.json")
OUTPUT_PATH = Path("data/tcgplayer_shadowless_map.json")


def normalize_name(name):
    """Normalize card name for matching."""
    name = name.lower().strip()
    # Remove parenthetical suffixes like "(3)" from TCGCSV names
    name = re.sub(r'\s*\(\d+\)\s*$', '', name)
    # Remove special chars
    name = re.sub(r'[^a-z0-9\s]', '', name)
    return name.strip()


def normalize_number(number):
    """Normalize card number: '002/102' -> '2', '58' -> '58'."""
    # Handle "NNN/NNN" format
    if '/' in number:
        number = number.split('/')[0]
    # Strip leading zeros
    return number.lstrip('0') or '0'


def fetch_shadowless_products():
    """Fetch all products from TCGCSV for Base Set (Shadowless)."""
    url = f"{TCGCSV_BASE}/{SHADOWLESS_GROUP_ID}/products"
    print(f"Fetching Shadowless products from {url}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    products = r.json().get("results", [])
    print(f"  Found {len(products)} Shadowless products")
    return products


def load_base1_cards():
    """Load base1 cards from card_index.json."""
    if not CARD_INDEX_PATH.exists():
        print(f"Error: {CARD_INDEX_PATH} not found.")
        print("Run 'python3 build_card_index.py' first.")
        sys.exit(1)

    with open(CARD_INDEX_PATH) as f:
        db = json.load(f)

    by_id = db.get("index", {}).get("by_id", {})
    base1_cards = {
        cid: card for cid, card in by_id.items()
        if cid.startswith("base1-")
    }
    print(f"  Found {len(base1_cards)} base1 cards in card_index.json")
    return base1_cards


def build_mapping(shadowless_products, base1_cards):
    """Match base1 cards to Shadowless product IDs by name + number."""
    # Index Shadowless products by normalized name + number
    shadowless_index = {}
    for p in shadowless_products:
        name = p.get("name", "")
        product_id = p.get("productId")
        if not product_id:
            continue

        # Get card number from extendedData
        number = ""
        for ed in p.get("extendedData", []):
            if ed.get("name") == "Number":
                number = ed.get("value", "")
                break

        norm_name = normalize_name(name)
        norm_num = normalize_number(number)
        key = f"{norm_name}/{norm_num}"
        shadowless_index[key] = {
            "product_id": product_id,
            "name": name,
            "number": number,
        }
        # Also index by number only for fallback
        if norm_num not in shadowless_index:
            shadowless_index[f"__num__{norm_num}"] = {
                "product_id": product_id,
                "name": name,
                "number": number,
            }

    mapping = {}
    matched = 0
    unmatched = []

    for card_id, card in sorted(base1_cards.items()):
        card_name = card.get("name", "")
        card_number = card.get("number", "")
        norm_name = normalize_name(card_name)
        norm_num = normalize_number(card_number)

        # Primary: name + number match
        key = f"{norm_name}/{norm_num}"
        if key in shadowless_index:
            entry = shadowless_index[key]
            mapping[card_id] = {
                "tcgplayer_id": entry["product_id"],
                "name": entry["name"],
                "confidence": "exact",
            }
            matched += 1
            continue

        # Fallback: number-only match
        num_key = f"__num__{norm_num}"
        if num_key in shadowless_index:
            entry = shadowless_index[num_key]
            mapping[card_id] = {
                "tcgplayer_id": entry["product_id"],
                "name": entry["name"],
                "confidence": "number_only",
            }
            matched += 1
            continue

        unmatched.append(f"  {card_id}: {card_name} #{card_number}")

    return mapping, matched, unmatched


def main():
    print("=" * 60)
    print("Building Base Set Shadowless TCGPlayer ID mapping")
    print("=" * 60)

    # Fetch data
    shadowless_products = fetch_shadowless_products()
    base1_cards = load_base1_cards()

    # Build mapping
    mapping, matched, unmatched = build_mapping(shadowless_products, base1_cards)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(mapping, f, indent=2)

    # Summary
    total = len(base1_cards)
    print()
    print(f"Results: {matched}/{total} cards mapped "
          f"({matched/total*100:.1f}%)")

    exact = sum(1 for v in mapping.values() if v["confidence"] == "exact")
    num_only = sum(1 for v in mapping.values() if v["confidence"] == "number_only")
    print(f"  Exact matches: {exact}")
    print(f"  Number-only:   {num_only}")

    if unmatched:
        print(f"\nUnmatched ({len(unmatched)}):")
        for u in unmatched:
            print(u)

    print(f"\nOutput: {OUTPUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()

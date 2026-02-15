"""
build_tcgplayer_map.py
Build a mapping from pokemontcg.io card IDs to TCGPlayer product IDs.

Downloads Pokemon product data from TCGCSV (free, no API key required),
then matches against card_index.json by card name + number within each set.

Output: tcgplayer_id_map.json
    {
        "base1-4": {"tcgplayer_id": 86142, "confidence": "exact"},
        "base1-58": {"tcgplayer_id": 86196, "confidence": "fuzzy"},
        ...
    }

Data sources:
    - TCGCSV (tcgcsv.com): TCGPlayer product data, updated daily, free
    - card_index.json: Local card database built from pokemon-tcg-data

Usage:
    python3 build_tcgplayer_map.py              # normal output
    python3 build_tcgplayer_map.py --verbose    # show per-card matching detail
    python3 build_tcgplayer_map.py --stats      # show only summary statistics
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

try:
    import requests
except ImportError:
    print("Missing dependency: pip install requests")
    sys.exit(1)

# ── Configuration ──

TCGCSV_BASE = "https://tcgcsv.com/tcgplayer"
POKEMON_CATEGORY_ID = 3
CARD_INDEX_PATH = Path("data/card_index.json")
OUTPUT_PATH = Path("data/tcgplayer_id_map.json")

# Minimum similarity ratio for fuzzy name matching (0.0–1.0)
FUZZY_THRESHOLD = 0.80

# ── Set Name Mapping ──
# Maps pokemontcg.io set IDs to likely TCGPlayer group names.
# Only needed where names diverge significantly; the matcher
# also tries normalized substring matching for unmapped sets.

# Direct pokemontcg.io set ID → TCGPlayer groupId mapping.
# Using groupIds eliminates false name matches entirely.
# To find missing groupIds: https://tcgcsv.com/tcgplayer/3/groups
POKEMONTCG_TO_TCGPLAYER_GROUP = {
    # WOTC era
    "base1": 604,       # Base Set
    "base2": 635,       # Jungle
    "basep": 1437,      # Wizards Black Star Promos
    "base3": 630,       # Fossil
    "base4": 605,       # Base Set 2
    "base5": 1373,      # Team Rocket
    "gym1": 1441,       # Gym Heroes
    "gym2": 1440,       # Gym Challenge
    "neo1": 1396,       # Neo Genesis
    "neo2": 1434,       # Neo Discovery
    "si1": 648,         # Southern Islands
    "neo3": 1389,       # Neo Revelation
    "neo4": 1444,       # Neo Destiny
    "base6": 1374,      # Legendary Collection
    "ecard1": 1418,     # Expedition Base Set
    "ecard2": 1397,     # Aquapolis
    "ecard3": 1372,     # Skyridge
    # EX era
    "ex1": 1393,        # Ruby and Sapphire
    "ex2": 1392,        # Sandstorm
    "ex3": 1376,        # Dragon
    "ex4": 1377,        # Team Magma vs Team Aqua
    "ex5": 1416,        # Hidden Legends
    "ex6": 1419,        # FireRed & LeafGreen
    "ex7": 1428,        # Team Rocket Returns
    "ex8": 1404,        # Deoxys
    "ex9": 1410,        # Emerald
    "ex10": 1398,       # Unseen Forces
    "ex11": 1429,       # Delta Species
    "ex12": 1378,       # Legend Maker
    "ex13": 1379,       # Holon Phantoms
    "ex14": 1395,       # Crystal Guardians
    "ex15": 1411,       # Dragon Frontiers
    "ex16": 1383,       # Power Keepers
    # DP era
    "dp1": 1430,        # Diamond and Pearl
    "dp2": 1368,        # Mysterious Treasures
    "dp3": 1380,        # Secret Wonders
    "dp4": 1405,        # Great Encounters
    "dp5": 1390,        # Majestic Dawn
    "dp6": 1417,        # Legends Awakened
    "dp7": 1369,        # Stormfront
    "dpp": 1421,        # Diamond and Pearl Promos
    # Platinum era
    "pl1": 1406,        # Platinum
    "pl2": 1367,        # Rising Rivals
    "pl3": 1384,        # Supreme Victors
    "pl4": 1391,        # Arceus
    # HGSS era
    "hgss1": 1402,      # HeartGold SoulSilver
    "hgss2": 1399,      # Unleashed
    "hgss3": 1403,      # Undaunted
    "hgss4": 1381,      # Triumphant
    "hsp": 1453,        # HGSS Promos
    # BW era
    "bw1": 1400,        # Black and White
    "bw2": 1424,        # Emerging Powers
    "bw3": 1385,        # Noble Victories
    "bw4": 1412,        # Next Destinies
    "bw5": 1386,        # Dark Explorers
    "bw6": 1394,        # Dragons Exalted
    "bw7": 1408,        # Boundaries Crossed
    "bw8": 1413,        # Plasma Storm
    "bw9": 1382,        # Plasma Freeze
    "bw10": 1370,       # Plasma Blast
    "bw11": 1409,       # Legendary Treasures
    "bwp": 1407,        # Black and White Promos
    # XY era
    "xy0": 1522,        # Kalos Starter Set
    "xy1": 1387,        # XY Base Set
    "xy2": 1464,        # XY - Flashfire
    "xy3": 1481,        # XY - Furious Fists
    "xy4": 1494,        # XY - Phantom Forces
    "xy5": 1509,        # XY - Primal Clash
    "xy6": 1534,        # XY - Roaring Skies
    "xy7": 1576,        # XY - Ancient Origins
    "xy8": 1661,        # XY - BREAKthrough
    "xy9": 1701,        # XY - BREAKpoint
    "xy10": 1780,       # XY - Fates Collide
    "xy11": 1815,       # XY - Steam Siege
    "xy12": 1842,       # XY - Evolutions
    "xyp": 1451,        # XY Promos
    "g1": 1728,         # Generations
    # SM era
    "sm1": 1863,        # SM Base Set
    "sm2": 1919,        # SM - Guardians Rising
    "sm3": 1957,        # SM - Burning Shadows
    "sm35": 2054,       # Shining Legends
    "sm4": 2071,        # SM - Crimson Invasion
    "sm5": 2178,        # SM - Ultra Prism
    "sm6": 2209,        # SM - Forbidden Light
    "sm7": 2278,        # SM - Celestial Storm
    "sm75": 2295,       # Dragon Majesty
    "sm8": 2328,        # SM - Lost Thunder
    "sm9": 2377,        # SM - Team Up
    "sm10": 2420,       # SM - Unbroken Bonds
    "sm11": 2464,       # SM - Unified Minds
    "sm115": 2480,      # Hidden Fates
    "sma": 2594,        # Hidden Fates: Shiny Vault
    "sm12": 2534,       # SM - Cosmic Eclipse
    "smp": 1861,        # SM Promos
    # SWSH era
    "swsh1": 2585,      # SWSH01: Sword & Shield Base Set
    "swsh2": 2626,      # SWSH02: Rebel Clash
    "swsh3": 2675,      # SWSH03: Darkness Ablaze
    "swsh35": 2685,     # Champion's Path
    "swsh4": 2701,      # SWSH04: Vivid Voltage
    "swsh45": 2754,     # Shining Fates
    "swsh45sv": 2781,   # Shining Fates: Shiny Vault
    "swsh5": 2765,      # SWSH05: Battle Styles
    "swsh6": 2807,      # SWSH06: Chilling Reign
    "swsh7": 2848,      # SWSH07: Evolving Skies
    "swsh8": 2906,      # SWSH08: Fusion Strike
    "swsh9": 2948,      # SWSH09: Brilliant Stars
    "swsh9tg": 3020,    # SWSH09: Brilliant Stars Trainer Gallery
    "swsh10": 3040,     # SWSH10: Astral Radiance
    "swsh10tg": 3068,   # SWSH10: Astral Radiance Trainer Gallery
    "swsh11": 3118,     # SWSH11: Lost Origin
    "swsh11tg": 3172,   # SWSH11: Lost Origin Trainer Gallery
    "swsh12": 3170,     # SWSH12: Silver Tempest
    "swsh12tg": 17674,  # SWSH12: Silver Tempest Trainer Gallery
    "swsh12pt5": 17688, # Crown Zenith
    "swsh12pt5gg": 17689, # Crown Zenith: Galarian Gallery
    "swshp": 2545,      # SWSH: Sword & Shield Promo Cards
    # SV era
    "sv1": 22873,       # SV01: Scarlet & Violet Base Set
    "sv2": 23120,       # SV02: Paldea Evolved
    "sv3": 23228,       # SV03: Obsidian Flames
    "sv3pt5": 23237,    # SV: Scarlet & Violet 151
    "sv4": 23286,       # SV04: Paradox Rift
    "sv4pt5": 23353,    # SV: Paldean Fates
    "sv5": 23381,       # SV05: Temporal Forces
    "sv6": 23473,       # SV06: Twilight Masquerade
    "sv6pt5": 23529,    # SV: Shrouded Fable
    "sv7": 23537,       # SV07: Stellar Crown
    "sv8": 23651,       # SV08: Surging Sparks
    "sv8pt5": 23821,    # SV: Prismatic Evolutions
    "sv9": 24073,       # SV09: Journey Together
    "sv10": 24269,      # SV10: Destined Rivals
    "svp": 22872,       # SV: Scarlet & Violet Promo Cards
    "sve": 24382,       # SVE: Scarlet & Violet Energies
    # Misc
    "mcd21": 2782,      # McDonald's 25th Anniversary Promos
    "ru1": 1433,        # Rumble
    "col1": 1415,       # Call of Legends
    "np": 1423,         # Nintendo Promos (groupId may vary — verify)
    "cel25": 2867,      # Celebrations
    "cel25c": 2931,     # Celebrations: Classic Collection
    "pgo": 3064,        # Pokemon GO
}


def normalize_name(name):
    """Normalize a card name for matching.
    Lowercases, strips accents, removes punctuation, collapses whitespace.
    """
    if not name:
        return ""
    # Unicode normalize → strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower().strip()
    # Remove common Pokemon-specific suffixes that differ between sources
    # e.g., "Charizard ex" vs "Charizard-ex"
    name = name.replace("-", " ").replace("'", "").replace("'", "")
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name)
    return name


def extract_card_number(number_str):
    """Extract just the card number from strings like '139/195' or '4'.
    Returns the number part as a string, stripped of leading zeros.
    """
    if not number_str:
        return ""
    # Handle "139/195" format
    match = re.match(r"(\d+)", str(number_str))
    if match:
        return str(int(match.group(1)))  # strip leading zeros
    return str(number_str).strip()


def normalize_set_name(name):
    """Normalize a set name for comparison."""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove common prefixes/suffixes
    for prefix in ["swsh", "sv", "sm", "xy", "bw", "dp", "ex", "hs"]:
        if name.startswith(prefix + ":"):
            name = name[len(prefix) + 1:].strip()
        elif name.startswith(prefix + " "):
            name = name[len(prefix) + 1:].strip()
    # Remove punctuation
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def load_card_index():
    """Load and parse card_index.json."""
    if not CARD_INDEX_PATH.exists():
        print(f"ERROR: {CARD_INDEX_PATH} not found. Run build_card_index.py first.")
        sys.exit(1)

    with open(CARD_INDEX_PATH) as f:
        data = json.load(f)

    index = data.get("index", {})
    by_id = index.get("by_id", {})
    print(f"  Loaded {len(by_id)} cards from {CARD_INDEX_PATH}")
    return index, by_id


def fetch_tcgcsv_groups():
    """Fetch all Pokemon groups (sets) from TCGCSV."""
    url = f"{TCGCSV_BASE}/{POKEMON_CATEGORY_ID}/groups"
    print(f"  Fetching groups from {url}...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    groups = data.get("results", [])
    print(f"  Found {len(groups)} Pokemon groups (sets)")
    return groups


def fetch_tcgcsv_products(group_id):
    """Fetch all products for a given group from TCGCSV."""
    url = f"{TCGCSV_BASE}/{POKEMON_CATEGORY_ID}/{group_id}/products"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])


def parse_tcgcsv_product(product):
    """Extract card-relevant fields from a TCGCSV product object.
    Returns dict with name, number, productId, or None if not a card.
    """
    extended = product.get("extendedData", [])
    ext_map = {e["name"]: e["value"] for e in extended}

    # Cards have a "Number" field in extendedData; sealed products don't
    number_raw = ext_map.get("Number", "")
    if not number_raw:
        return None

    card_number = extract_card_number(number_raw)
    if not card_number:
        return None

    return {
        "product_id": product["productId"],
        "name": product.get("cleanName", product.get("name", "")),
        "number": card_number,
        "rarity": ext_map.get("Rarity", ""),
    }


def match_cards(pokemontcg_cards, tcg_products, verbose=False):
    """Match pokemontcg.io cards to TCGPlayer products within a single set.

    Strategy:
      1. Exact match on card number + exact normalized name → confidence "exact"
      2. Exact match on card number + fuzzy name (≥80% similarity) → "fuzzy"
      3. If only one product matches the number → "number_only"

    Returns list of (pokemontcg_id, tcgplayer_product_id, confidence) tuples.
    """
    matches = []

    # Index TCG products by card number for fast lookup
    products_by_number = {}
    for p in tcg_products:
        num = p["number"]
        if num not in products_by_number:
            products_by_number[num] = []
        products_by_number[num].append(p)

    for card_id, card in pokemontcg_cards.items():
        card_num = extract_card_number(card.get("number", ""))
        if not card_num:
            continue

        card_name_norm = normalize_name(card.get("name", ""))
        candidates = products_by_number.get(card_num, [])

        if not candidates:
            continue

        # Try exact name match first
        best_match = None
        best_confidence = None
        best_ratio = 0.0

        for p in candidates:
            p_name_norm = normalize_name(p["name"])

            if card_name_norm == p_name_norm:
                best_match = p
                best_confidence = "exact"
                best_ratio = 1.0
                break

            ratio = SequenceMatcher(None, card_name_norm, p_name_norm).ratio()
            if ratio >= FUZZY_THRESHOLD and ratio > best_ratio:
                best_match = p
                best_confidence = "fuzzy"
                best_ratio = ratio

        # Fallback: if only one product has this number, use it
        if not best_match and len(candidates) == 1:
            best_match = candidates[0]
            best_confidence = "number_only"
            best_ratio = 0.0

        if best_match:
            matches.append((card_id, best_match["product_id"], best_confidence))
            if verbose:
                print(f"    {card_id} → {best_match['product_id']} "
                      f"({best_confidence}, ratio={best_ratio:.2f}) "
                      f"'{card.get('name', '')}' ↔ '{best_match['name']}'")
        elif verbose:
            print(f"    {card_id} → NO MATCH "
                  f"(name='{card.get('name', '')}', num={card_num}, "
                  f"candidates={len(candidates)})")

    return matches


def build_set_mapping(by_id):
    """Group pokemontcg.io cards by their set ID."""
    sets = {}
    for card_id, card in by_id.items():
        set_data = card.get("set", {})
        set_id = set_data.get("id", "")
        if set_id:
            if set_id not in sets:
                sets[set_id] = {"name": set_data.get("name", ""), "cards": {}}
            sets[set_id]["cards"][card_id] = card
    return sets


def find_matching_group(set_id, set_name, tcg_groups):
    """Find the TCGPlayer group that matches a pokemontcg.io set.

    Tries:
      1. Direct groupId from mapping table (most reliable)
      2. Exact normalized name match
      3. Fuzzy name match (≥0.70 threshold)
    """
    # Index groups by ID for O(1) lookup
    groups_by_id = {g["groupId"]: g for g in tcg_groups}

    # 1. Direct groupId mapping (most reliable)
    mapped_group_id = POKEMONTCG_TO_TCGPLAYER_GROUP.get(set_id)
    if mapped_group_id and mapped_group_id in groups_by_id:
        return groups_by_id[mapped_group_id]

    # 2. Exact normalized name match
    set_norm = normalize_set_name(set_name)
    for g in tcg_groups:
        if normalize_set_name(g["name"]) == set_norm:
            return g
        # Also try after stripping era prefix (e.g., "SWSH12: Silver Tempest")
        if ":" in g["name"]:
            g_suffix = g["name"].split(":", 1)[1].strip()
            if normalize_set_name(g_suffix) == set_norm:
                return g

    # 3. Fuzzy match (but require high threshold)
    best_group = None
    best_ratio = 0.0
    for g in tcg_groups:
        g_norm = normalize_set_name(g["name"])
        ratio = SequenceMatcher(None, set_norm, g_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_group = g
        if ":" in g["name"]:
            g_suffix = normalize_set_name(g["name"].split(":", 1)[1].strip())
            ratio2 = SequenceMatcher(None, set_norm, g_suffix).ratio()
            if ratio2 > best_ratio:
                best_ratio = ratio2
                best_group = g

    if best_ratio >= 0.70:
        return best_group

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Build pokemontcg.io → TCGPlayer ID mapping"
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-card matching detail")
    parser.add_argument("--stats", "-s", action="store_true",
                        help="Show only summary statistics")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("BUILD TCGPLAYER ID MAP")
    print("=" * 60)

    # Load local card index
    print("\n[1/4] Loading card index...")
    card_index, by_id = load_card_index()
    pokemontcg_sets = build_set_mapping(by_id)
    print(f"  {len(pokemontcg_sets)} sets found in card index")

    # Fetch TCGCSV groups
    print("\n[2/4] Fetching TCGPlayer groups from TCGCSV...")
    tcg_groups = fetch_tcgcsv_groups()

    # Match sets and fetch products
    print("\n[3/4] Matching sets and fetching products...")
    all_matches = []
    sets_matched = 0
    sets_unmatched = []
    sets_processed = 0

    for set_id, set_data in sorted(pokemontcg_sets.items()):
        set_name = set_data["name"]
        card_count = len(set_data["cards"])
        sets_processed += 1

        group = find_matching_group(set_id, set_name, tcg_groups)
        if not group:
            sets_unmatched.append((set_id, set_name, card_count))
            if not args.stats:
                print(f"  [{sets_processed}/{len(pokemontcg_sets)}] "
                      f"{set_id} ({set_name}): NO GROUP MATCH")
            continue

        sets_matched += 1

        # Fetch products for this group
        try:
            products = fetch_tcgcsv_products(group["groupId"])
        except Exception as e:
            if not args.stats:
                print(f"  [{sets_processed}/{len(pokemontcg_sets)}] "
                      f"{set_id} ({set_name}): FETCH ERROR: {e}")
            continue

        # Parse card products (skip sealed products)
        tcg_cards = []
        for p in products:
            parsed = parse_tcgcsv_product(p)
            if parsed:
                tcg_cards.append(parsed)

        # Match cards
        matches = match_cards(set_data["cards"], tcg_cards, verbose=args.verbose)
        all_matches.extend(matches)

        if not args.stats:
            print(f"  [{sets_processed}/{len(pokemontcg_sets)}] "
                  f"{set_id} → {group['name']} (group {group['groupId']}): "
                  f"{len(matches)}/{card_count} cards matched, "
                  f"{len(tcg_cards)} TCG products")

        # Be polite to TCGCSV
        time.sleep(0.1)

    # Build output map
    print(f"\n[4/4] Writing {OUTPUT_PATH}...")
    id_map = {}
    confidence_counts = {"exact": 0, "fuzzy": 0, "number_only": 0}

    for card_id, product_id, confidence in all_matches:
        id_map[card_id] = {
            "tcgplayer_id": product_id,
            "confidence": confidence,
        }
        confidence_counts[confidence] += 1

    with open(OUTPUT_PATH, "w") as f:
        json.dump(id_map, f, indent=2)

    # Summary
    elapsed = time.time() - start_time
    total_cards = len(by_id)
    mapped_cards = len(id_map)
    coverage = (mapped_cards / total_cards * 100) if total_cards else 0

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total cards in index:   {total_cards}")
    print(f"  Cards mapped:           {mapped_cards} ({coverage:.1f}%)")
    print(f"  Exact name matches:     {confidence_counts['exact']}")
    print(f"  Fuzzy name matches:     {confidence_counts['fuzzy']}")
    print(f"  Number-only matches:    {confidence_counts['number_only']}")
    print(f"  Sets matched:           {sets_matched}/{len(pokemontcg_sets)}")
    print(f"  Sets unmatched:         {len(sets_unmatched)}")
    print(f"  Output:                 {OUTPUT_PATH}")
    print(f"  Time:                   {elapsed:.1f}s")

    if sets_unmatched and not args.stats:
        print(f"\n  Unmatched sets:")
        for sid, sname, cnt in sets_unmatched:
            print(f"    {sid}: {sname} ({cnt} cards)")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

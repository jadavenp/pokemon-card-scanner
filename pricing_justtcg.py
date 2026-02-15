"""
pricing_justtcg.py
Live card pricing via JustTCG API.

Returns per-condition pricing (Near Mint default) with support for
1st Edition printing lookups.

Requires:
    - data/tcgplayer_id_map.json (built by build_tcgplayer_map.py)
    - data/tcgplayer_shadowless_map.json (built by build_shadowless_map.py)
    - .env with JUSTTCG_API_KEY

Return format for fetch_live_pricing():
    {"price": "$XX.XX", "variant": "1st_ed_nm", "source": "JustTCG"}

Drop-in replacement for pricing.py — same function signature.

1st Edition pricing strategy:
  - Base Set (base1): Uses separate Shadowless product IDs (TCGPlayer group 1663)
    because TCGPlayer stores Base Set 1st Ed under "Base Set (Shadowless)".
  - Other WOTC sets (Jungle, Fossil, Team Rocket, etc.): Same product ID,
    but filtered by printing="1st Edition" or "1st Edition Holofoil".
  - Holo vs non-holo determined by rarity field from card_index.json.
"""

import json
import os
import requests
from pathlib import Path

import config  # triggers .env loading from config.py's custom parser

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env already loaded by config.py

# ── Configuration ──

JUSTTCG_BASE = "https://api.justtcg.com/v1"
JUSTTCG_API_KEY = os.getenv("JUSTTCG_API_KEY", "")
JUSTTCG_HEADERS = {
    "x-api-key": JUSTTCG_API_KEY,
}

TCGPLAYER_MAP_PATH = Path("data/tcgplayer_id_map.json")
SHADOWLESS_MAP_PATH = Path("data/tcgplayer_shadowless_map.json")

# Module-level caches
_tcgplayer_map = None
_shadowless_map = None


# ── Base Set Shadowless mapping ──
# Base Set (base1) 1st Edition cards live under TCGPlayer group 1663
# ("Base Set (Shadowless)") with separate product IDs. Other WOTC sets
# keep 1st Edition as a printing variant under the same product ID.

SETS_USING_SHADOWLESS = {"base1"}

# Manual overrides for cards with non-standard TCGPlayer listings.
# Machamp (base1-8) is listed under TCGPlayer "Deck Exclusives" (product 107004),
# not in Base Set (Shadowless) group 1663, because it was only available as
# 1st Edition in the starter deck.
SHADOWLESS_OVERRIDES = {
    "base1-8": 107004,  # Machamp - Deck Exclusives
}


def _load_tcgplayer_map():
    """Load the pokemontcg_id -> tcgplayerId mapping. Cached after first call."""
    global _tcgplayer_map
    if _tcgplayer_map is not None:
        return _tcgplayer_map

    if not TCGPLAYER_MAP_PATH.exists():
        print(f"  WARNING: {TCGPLAYER_MAP_PATH} not found. "
              f"Run build_tcgplayer_map.py to enable JustTCG pricing.")
        _tcgplayer_map = {}
        return _tcgplayer_map

    with open(TCGPLAYER_MAP_PATH) as f:
        _tcgplayer_map = json.load(f)
    return _tcgplayer_map


def _load_shadowless_map():
    """Load the Base Set Shadowless mapping. Cached after first call."""
    global _shadowless_map
    if _shadowless_map is not None:
        return _shadowless_map

    if not SHADOWLESS_MAP_PATH.exists():
        # Not fatal — we'll fall back to regular map
        _shadowless_map = {}
        return _shadowless_map

    with open(SHADOWLESS_MAP_PATH) as f:
        _shadowless_map = json.load(f)
    return _shadowless_map


def _get_set_id(card_id):
    """Extract set ID from card_id (e.g., 'base1' from 'base1-58')."""
    parts = card_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else ""


def _is_holo(rarity):
    """Determine if a card is holo based on rarity string.
    JustTCG uses 'Holofoil' suffix for holo printings:
      - Holo: '1st Edition Holofoil', 'Unlimited Holofoil'
      - Non-holo: '1st Edition', 'Unlimited', 'Normal'
    """
    if not rarity:
        return False
    return "holo" in rarity.lower()


def _get_tcgplayer_id(card_id, is_1st_edition=False):
    """Look up the TCGPlayer product ID for a pokemontcg.io card ID.

    For Base Set 1st Edition cards, returns the Shadowless product ID
    (TCGPlayer group 1663) since that's where 1st Ed pricing lives.

    Returns the product ID as a string, or None if not found.
    """
    set_id = _get_set_id(card_id)

    # Manual overrides for cards with non-standard TCGPlayer listings
    if is_1st_edition and card_id in SHADOWLESS_OVERRIDES:
        return str(SHADOWLESS_OVERRIDES[card_id])

    # Base Set 1st Edition -> use Shadowless product IDs
    if is_1st_edition and set_id in SETS_USING_SHADOWLESS:
        shadowless_map = _load_shadowless_map()
        entry = shadowless_map.get(card_id)
        if entry:
            return str(entry["tcgplayer_id"])
        # Fall through to regular map if shadowless mapping missing

    id_map = _load_tcgplayer_map()
    entry = id_map.get(card_id)
    if entry:
        return str(entry["tcgplayer_id"])
    return None


def _determine_printing_filter(is_1st_edition, rarity, set_id):
    """Determine the JustTCG printing parameter value.

    Returns the printing string to filter by, or None for no filter.

    JustTCG printing values observed:
      Non-holo: 'Normal', '1st Edition', 'Unlimited'
      Holo:     'Holofoil', '1st Edition Holofoil', 'Unlimited Holofoil'

    For Base Set (Shadowless) products:
      - The Shadowless product already separates 1st Ed / Unlimited
      - Use '1st Edition' / '1st Edition Holofoil' to filter

    For other WOTC sets:
      - Same product has both printings
      - Use '1st Edition' / '1st Edition Holofoil' to filter
    """
    if not is_1st_edition:
        # Don't filter — let JustTCG return Normal/Holofoil/Unlimited
        return None

    holo = _is_holo(rarity)
    if holo:
        return "1st Edition Holofoil"
    else:
        return "1st Edition"


def fetch_justtcg(card_id, is_1st_edition=False, rarity="Common",
                  condition="NM"):
    """
    Fetch pricing from JustTCG API.

    Args:
        card_id: pokemontcg.io card ID (e.g., "base1-58")
        is_1st_edition: If True, filter for 1st Edition printing
        rarity: Card rarity string (e.g., "Rare Holo", "Rare", "Common")
        condition: Condition abbreviation (NM, LP, MP, HP, DMG)

    Returns (price_str, variant_str, source) or (None, None, None) on failure.
    """
    tcgplayer_id = _get_tcgplayer_id(card_id, is_1st_edition=is_1st_edition)
    if not tcgplayer_id:
        return None, None, None

    if not JUSTTCG_API_KEY:
        return None, None, None

    set_id = _get_set_id(card_id)

    try:
        params = {
            "tcgplayerId": tcgplayer_id,
            "game": "pokemon",
            "include_price_history": "false",
            "include_statistics": "",
        }

        # Add printing filter for 1st Edition
        printing_filter = _determine_printing_filter(
            is_1st_edition, rarity, set_id
        )
        if printing_filter:
            params["printing"] = printing_filter

        r = requests.get(
            f"{JUSTTCG_BASE}/cards",
            headers=JUSTTCG_HEADERS,
            params=params,
            timeout=10,
        )

        if r.status_code == 200:
            data = r.json()
            cards = data.get("data", [])

            if cards and isinstance(cards, list):
                card = cards[0]
                variants = card.get("variants", [])

                if variants:
                    best = _select_best_variant(
                        variants, is_1st_edition, condition
                    )
                    if best:
                        price = best.get("price")
                        if price is not None:
                            printing = best.get("printing", "Normal")
                            cond = best.get("condition", condition)
                            variant_label = _build_variant_label(
                                printing, cond, is_1st_edition
                            )
                            return f"${price:.2f}", variant_label, "JustTCG"

            # If printing filter returned no results, retry without filter
            # (fallback to Unlimited pricing with a note)
            if not cards and printing_filter:
                params.pop("printing", None)
                r2 = requests.get(
                    f"{JUSTTCG_BASE}/cards",
                    headers=JUSTTCG_HEADERS,
                    params=params,
                    timeout=10,
                )
                if r2.status_code == 200:
                    data2 = r2.json()
                    cards2 = data2.get("data", [])
                    if cards2:
                        card2 = cards2[0]
                        variants2 = card2.get("variants", [])
                        if variants2:
                            best = _select_best_variant(
                                variants2, False, condition
                            )
                            if best:
                                price = best.get("price")
                                if price is not None:
                                    cond = best.get("condition", condition)
                                    variant_label = _build_variant_label(
                                        best.get("printing", "Normal"),
                                        cond, is_1st_edition
                                    )
                                    return (f"${price:.2f}", variant_label,
                                            "JustTCG")

        elif r.status_code == 429:
            return None, None, None

    except Exception:
        pass

    return None, None, None


def _select_best_variant(variants, is_1st_edition, target_condition):
    """Select the best variant from JustTCG results.

    Priority:
      1. Exact printing + condition match
      2. Right printing, any condition (prefer NM -> LP -> MP)
      3. Any variant with a price
    """
    condition_priority = ["Near Mint", "Lightly Played", "Moderately Played",
                          "Heavily Played", "Damaged"]
    condition_abbrev = {
        "NM": "Near Mint", "LP": "Lightly Played", "MP": "Moderately Played",
        "HP": "Heavily Played", "DMG": "Damaged",
    }
    target_full = condition_abbrev.get(target_condition, target_condition)
    target_printing = "1st Edition" if is_1st_edition else None

    # Filter to variants with valid prices
    priced = [v for v in variants if v.get("price") is not None and v["price"] > 0]
    if not priced:
        return None

    # If looking for 1st Edition, try to find it
    if target_printing:
        printing_match = [
            v for v in priced
            if target_printing.lower() in v.get("printing", "").lower()
        ]
        if printing_match:
            # Exact condition match within printing
            for v in printing_match:
                if v.get("condition", "") == target_full:
                    return v
            # Best available condition within printing
            for cond in condition_priority:
                for v in printing_match:
                    if v.get("condition", "") == cond:
                        return v
            return printing_match[0]

    # For unlimited or if 1st Ed not found: match by condition
    for v in priced:
        if v.get("condition", "") == target_full:
            printing = v.get("printing", "")
            p_lower = printing.lower()
            if not target_printing and p_lower in (
                "normal", "holofoil", "unlimited", "unlimited holofoil"
            ):
                return v

    # Fallback: best condition available
    for cond in condition_priority:
        for v in priced:
            if v.get("condition", "") == cond:
                return v

    return priced[0] if priced else None


def _build_variant_label(printing, condition, is_1st_edition):
    """Build a human-readable variant label.
    Examples: '1st_ed_nm', 'unlimited_nm', 'holofoil_lp'
    """
    cond_short = {
        "Near Mint": "nm", "Lightly Played": "lp",
        "Moderately Played": "mp", "Heavily Played": "hp",
        "Damaged": "dmg", "Sealed": "sealed",
    }
    cond = cond_short.get(condition, condition.lower())

    if is_1st_edition and "1st" in printing.lower():
        return f"1st_ed_{cond}"
    elif is_1st_edition:
        # 1st Ed was requested but not found in results
        return f"unlimited*_{cond}"

    printing_short = printing.lower().replace(" ", "_") if printing else "normal"
    if printing_short in ("normal", "holofoil", "unlimited",
                          "unlimited_holofoil"):
        return f"unlimited_{cond}"
    return f"{printing_short}_{cond}"


def fetch_live_pricing(card_id, is_1st_edition=False, rarity="Common"):
    """
    Fetch pricing for a card via JustTCG.

    Args:
        card_id: Card ID (e.g., "base1-58")
        is_1st_edition: Whether stamp detection flagged 1st Edition
        rarity: Card rarity string from database (e.g., "Rare Holo")

    Returns dict:
        price:   "$XX.XX" or "N/A"
        variant: "1st_ed_nm", "unlimited_nm", etc.
        source:  "JustTCG" or ""

    1st Edition pricing strategy:
      - Base Set: Looks up Shadowless product ID (separate TCGPlayer listing)
      - Other WOTC sets: Same product ID, filtered by printing param
      - Holo detection via rarity field selects correct printing value
    """
    price, variant, source = fetch_justtcg(
        card_id, is_1st_edition=is_1st_edition, rarity=rarity
    )
    if price:
        return {"price": price, "variant": variant, "source": source}

    return {"price": "N/A", "variant": "", "source": ""}

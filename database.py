"""
database.py — Local card database loading and lookup.
Database: card_index.json (20,078 cards).

Refactored (Session 4) to add multi_signal_lookup() which cross-references
multiple OCR signals (name candidates, HP, number, set total, card type)
against the database to find the best match. This replaces the old
single-name-winner approach that was prone to misidentification.

Existing functions (lookup_card, lookup_by_name, build_set_totals) are
preserved for backward compatibility.
"""

import json
import sys
import logging

from config import DATABASE_FILE, VINTAGE_SET_IDS

logger = logging.getLogger("database")


def load_database():
    """
    Load the card database and return (index_dict, card_count).
    Exits if the database file is missing.

    The index_dict contains:
      - by_id: card_id -> card dict
      - by_name: lowercase name -> [card_id, ...]
      - by_number_and_name: "number/name" -> [card_id, ...]
      - by_set_number: "set_id/number" -> card dict
    """
    if not DATABASE_FILE.exists():
        print(f"Error: Database not found at {DATABASE_FILE}")
        print("Run 'python3 build_card_index.py' first.")
        sys.exit(1)

    with open(DATABASE_FILE, "r") as f:
        database = json.load(f)

    index = database.get("index", {})
    card_count = database.get("meta", {}).get("card_count", 0)
    return index, card_count


def get_set_id(card):
    """Extract set ID from a card dict or card ID string."""
    if isinstance(card, dict):
        set_obj = card.get("set", {})
        if isinstance(set_obj, dict) and set_obj.get("id"):
            return set_obj["id"]
        card_id = card.get("id", "")
    else:
        card_id = str(card)

    parts = card_id.rsplit("-", 1)
    return parts[0] if len(parts) == 2 else ""


def _resolve_ids(index, card_ids):
    """Resolve a list of card IDs to full card dicts using by_id."""
    by_id = index.get("by_id", {})
    results = []
    for cid in card_ids:
        if isinstance(cid, dict):
            results.append(cid)
        elif isinstance(cid, str) and cid in by_id:
            results.append(by_id[cid])
    return results


# ─────────────────────────────────────────────────────────────
# MULTI-SIGNAL LOOKUP (NEW)
# ─────────────────────────────────────────────────────────────

# Signal weights for composite scoring
_SIGNAL_WEIGHTS = {
    "name_exact": 0.30,       # Exact name match in by_name index
    "name_fuzzy": 0.15,       # Substring/partial name match
    "hp": 0.20,               # HP matches card's hp field
    "number": 0.20,           # Card number matches
    "set_total": 0.10,        # Set total matches set.printedTotal
    "card_type": 0.05,        # Supertype matches (pokemon/trainer/energy)
}


def multi_signal_lookup(index, name_candidates=None, hp=None, number=None,
                        total=None, card_type=None, max_results=5):
    """
    Cross-reference multiple OCR signals against the database to find
    the best card match. Returns ranked candidates with composite scores.

    This replaces the old approach of:
      1. Pick ONE name winner
      2. Look up name + number
      3. Hope it's right

    With:
      1. Try ALL name candidates against DB
      2. For each name hit, score how many other signals match
      3. Return the top results ranked by composite agreement

    Args:
        index: card index database (from load_database())
        name_candidates: list of dicts from extract_name_candidates()
            Each has: {"name": str, "confidence": float, "score": float}
        hp: int or None — extracted HP value
        number: str or None — extracted card number (e.g. "55")
        total: str or None — extracted set total (e.g. "78")
        card_type: str or None — "pokemon", "trainer", "energy"
        max_results: max results to return

    Returns:
        list of dicts sorted by composite_score (best first):
        [
            {
                "card": {full card dict},
                "card_id": "pgo-55",
                "composite_score": 0.92,
                "signals_matched": {"name_exact": True, "hp": True, "number": True, ...},
                "match_count": 4,
                "name_source": "Snorlax",
                "name_confidence": 100.0,
                "ambiguous": False,
            },
            ...
        ]
    """
    if not name_candidates:
        name_candidates = []

    by_id = index.get("by_id", {})
    by_name = index.get("by_name", {})

    # Normalize inputs
    number_clean = number.lstrip("0") if number else None
    total_clean = total.lstrip("0") if total else None

    # Map supertype strings for comparison
    # DB stores "Pokémon" (with accent), "Trainer", "Energy"
    type_map = {
        "pokemon": ["pokémon", "pokemon"],
        "trainer": ["trainer"],
        "energy": ["energy"],
    }
    expected_supertypes = type_map.get(card_type, []) if card_type else []

    # ── Gather candidate cards from all name candidates ──
    seen_card_ids = set()
    scored_results = []

    for name_cand in name_candidates:
        cand_name = name_cand["name"]
        cand_conf = name_cand.get("confidence", 0)
        cand_name_lower = cand_name.lower().strip()

        # Find cards matching this name
        matching_cards = []

        # Exact name match
        exact_ids = by_name.get(cand_name_lower, [])
        exact_cards = _resolve_ids(index, exact_ids)
        name_match_type = "name_exact"

        if not exact_cards:
            # Fuzzy: substring match
            for name_key, card_ids in by_name.items():
                if cand_name_lower in name_key or name_key in cand_name_lower:
                    exact_cards.extend(_resolve_ids(index, card_ids))
            name_match_type = "name_fuzzy"

        for card in exact_cards:
            card_id = card.get("id", "")
            if card_id in seen_card_ids:
                continue
            seen_card_ids.add(card_id)

            # ── Score each signal ──
            signals = {}
            score = 0.0

            # Name signal
            signals[name_match_type] = True
            weight = _SIGNAL_WEIGHTS.get(name_match_type, 0.15)
            # Scale name weight by the OCR confidence of this candidate
            score += weight * (cand_conf / 100.0)

            # HP signal
            # DB stores HP as string (e.g. "150"), extract_hp returns int
            if hp is not None:
                card_hp_str = card.get("hp", "")
                if card_hp_str:
                    try:
                        card_hp_int = int(card_hp_str)
                        if card_hp_int == hp:
                            signals["hp"] = True
                            score += _SIGNAL_WEIGHTS["hp"]
                        else:
                            signals["hp"] = False
                            # Penalty: HP mismatch is a strong negative signal
                            score -= _SIGNAL_WEIGHTS["hp"] * 0.5
                    except (ValueError, TypeError):
                        pass  # Non-numeric HP field (shouldn't happen but be safe)

            # Number signal
            # DB stores number as string without leading zeros (e.g. "55")
            # OCR extracts "055" → lstrip("0") → "55"
            if number_clean is not None:
                card_number = (card.get("number") or "").lstrip("0") or "0"
                if card_number == number_clean:
                    signals["number"] = True
                    score += _SIGNAL_WEIGHTS["number"]
                else:
                    signals["number"] = False

            # Set total signal
            # DB stores printedTotal as int (e.g. 78), OCR gives string "78"
            if total_clean is not None:
                card_set = card.get("set", {})
                if isinstance(card_set, dict):
                    printed_total = card_set.get("printedTotal")
                    if printed_total is not None:
                        printed_total_str = str(printed_total).lstrip("0") or "0"
                        if printed_total_str == total_clean:
                            signals["set_total"] = True
                            score += _SIGNAL_WEIGHTS["set_total"]
                        else:
                            signals["set_total"] = False

            # Card type signal
            # DB stores "Pokémon" (with accent), "Trainer", "Energy"
            if expected_supertypes:
                card_supertype = card.get("supertype", "").lower()
                if any(st in card_supertype for st in expected_supertypes):
                    signals["card_type"] = True
                    score += _SIGNAL_WEIGHTS["card_type"]
                else:
                    signals["card_type"] = False

            match_count = sum(1 for v in signals.values() if v is True)

            scored_results.append({
                "card": card,
                "card_id": card_id,
                "composite_score": round(score, 3),
                "signals_matched": signals,
                "match_count": match_count,
                "name_source": cand_name,
                "name_confidence": cand_conf,
                "ambiguous": False,
            })

    # ── Sort by composite score, break ties by match count ──
    scored_results.sort(key=lambda x: (x["composite_score"], x["match_count"]), reverse=True)

    # ── Check for ambiguity ──
    # If top 2 results have very close scores, flag as ambiguous
    if len(scored_results) >= 2:
        top_score = scored_results[0]["composite_score"]
        second_score = scored_results[1]["composite_score"]
        if top_score > 0 and (top_score - second_score) / top_score < 0.10:
            scored_results[0]["ambiguous"] = True

    results = scored_results[:max_results]

    # Log the results
    if results:
        top = results[0]
        signals_str = ", ".join(
            f"{k}={'✓' if v else '✗'}" for k, v in top["signals_matched"].items()
        )
        logger.info(
            "Multi-signal lookup: %s (%s) score=%.3f matches=%d [%s]",
            top["card_id"], top["name_source"], top["composite_score"],
            top["match_count"], signals_str,
        )
        if top["ambiguous"]:
            logger.info("  ⚠ Ambiguous: top 2 scores within 10%%")
        if len(results) > 1:
            for r in results[1:3]:
                logger.info("  runner-up: %s (score=%.3f, matches=%d, name='%s')",
                            r["card_id"], r["composite_score"], r["match_count"],
                            r["name_source"])
    else:
        logger.info("Multi-signal lookup: no matches found")

    return results


# ─────────────────────────────────────────────────────────────
# EXISTING LOOKUP FUNCTIONS (preserved)
# ─────────────────────────────────────────────────────────────

def lookup_card(index, card_name, card_number):
    """
    Search the local database for a card by name and number.
    Uses three strategies in order:
      1. Exact match on number + lowercase name
      2. Name-only match, filtered by number
      3. Fuzzy substring match on name, filtered by number

    Returns list of matching card dicts (may be empty).
    """
    matches = []

    # Primary: number + name
    lookup_key = f"{card_number}/{card_name.lower()}"
    entries = index.get("by_number_and_name", {}).get(lookup_key, [])
    if entries:
        matches = _resolve_ids(index, entries)

    # Fallback: name only, filter by number
    if not matches:
        name_entries = index.get("by_name", {}).get(card_name.lower(), [])
        name_cards = _resolve_ids(index, name_entries)
        matches = [c for c in name_cards if c.get("number") == card_number]

    # Fuzzy fallback: substring match
    if not matches:
        search_name = card_name.lower()
        for name_key, entries in index.get("by_name", {}).items():
            if search_name in name_key or name_key in search_name:
                name_cards = _resolve_ids(index, entries)
                number_matches = [c for c in name_cards
                                  if c.get("number") == card_number]
                matches.extend(number_matches)

    return matches


def build_set_totals(index):
    """
    Build a mapping of set_total -> [set_id, ...] from the card database.

    Scans all cards to find the printedTotal per set (the number after "/"
    on each card, e.g. 055/078 → total="78"). This replaces hardcoded
    KNOWN_SET_TOTALS and is always complete and accurate.

    Returns dict like {"78": ["pgo"], "195": ["swsh12"], ...}
    """
    set_totals = {}  # set_id -> printed total string
    by_id = index.get("by_id", {})

    for card_id, card in by_id.items():
        set_id = get_set_id(card)
        if not set_id:
            continue
        set_obj = card.get("set", {})
        if isinstance(set_obj, dict):
            printed_total = set_obj.get("printedTotal") or set_obj.get("total")
            if printed_total:
                set_totals[set_id] = str(printed_total)

    # Invert: total_string -> [set_ids]
    totals_map = {}
    for set_id, total in set_totals.items():
        if total not in totals_map:
            totals_map[total] = []
        if set_id not in totals_map[total]:
            totals_map[total].append(set_id)

    return totals_map


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
        matches = _resolve_ids(index, name_matches)

    # Fuzzy fallback: substring match
    if not matches:
        for name_key, cards_list in index.get("by_name", {}).items():
            if search_name in name_key or name_key in search_name:
                matches.extend(_resolve_ids(index, cards_list))

    # Filter by set hint if provided
    if set_hint and matches:
        filtered = [c for c in matches if get_set_id(c) == set_hint]
        if filtered:
            matches = filtered

    # Sort: vintage sets first (higher collector value), then by number
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

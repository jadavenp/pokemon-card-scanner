"""
database.py
Local card database loading and lookup.
Database: card_index.json (20,078 cards).
"""

import json
import sys

from config import DATABASE_FILE


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
        # Try set.id first
        set_obj = card.get("set", {})
        if isinstance(set_obj, dict) and set_obj.get("id"):
            return set_obj["id"]
        # Fall back to parsing card ID
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
            # Already a full card dict (backward compat)
            results.append(cid)
        elif isinstance(cid, str) and cid in by_id:
            results.append(by_id[cid])
    return results


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

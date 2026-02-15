"""
test_pricing_e2e.py
End-to-end test for JustTCG pricing across all fix items.

Tests:
  1. Jungle/Fossil 1st Ed (base2/base3 IDs) — Item 1
  2. Machamp (base1-8) 1st Ed via override — Item 2
  3. Base Set cards via Shadowless map — existing functionality
  4. Variant labels appear in VARIANT_DISPLAY — Item 4

Usage:
    cd ~/card-scanner && source venv/bin/activate
    python3 test_pricing_e2e.py
"""

import sys
from config import VARIANT_DISPLAY
from pricing_justtcg import (
    fetch_live_pricing, _get_tcgplayer_id, SHADOWLESS_OVERRIDES,
    JUSTTCG_API_KEY,
)


def test_api_key():
    """Verify API key is loaded."""
    if not JUSTTCG_API_KEY:
        print("FAIL: JUSTTCG_API_KEY not loaded. Check .env file.")
        return False
    print(f"PASS: API key loaded (length: {len(JUSTTCG_API_KEY)})")
    return True


def test_machamp_override():
    """Item 2: Machamp override resolves to product 107004."""
    tcg_id = _get_tcgplayer_id("base1-8", is_1st_edition=True)
    expected = "107004"
    if tcg_id == expected:
        print(f"PASS: Machamp 1st Ed -> TCGPlayer {tcg_id} (Deck Exclusives)")
    else:
        print(f"FAIL: Machamp 1st Ed -> {tcg_id} (expected {expected})")
    return tcg_id == expected


def test_variant_display():
    """Item 4: JustTCG variant labels are in VARIANT_DISPLAY."""
    required = ["1st_ed_nm", "unlimited_nm", "unlimited*_nm",
                 "1st_ed_lp", "unlimited_lp", "holofoil_nm"]
    missing = [v for v in required if v not in VARIANT_DISPLAY]
    if not missing:
        print(f"PASS: All {len(required)} JustTCG variant labels present")
    else:
        print(f"FAIL: Missing variant labels: {missing}")
    return len(missing) == 0


def test_pricing(card_id, is_1st, rarity, label):
    """Test a single pricing call."""
    result = fetch_live_pricing(card_id, is_1st_edition=is_1st, rarity=rarity)
    price = result["price"]
    variant = result["variant"]
    source = result["source"]
    ok = price != "N/A"
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {label:30s} -> {price:>10s} [{variant}] ({source})")
    return ok


def main():
    print("=" * 65)
    print("JustTCG Pricing — End-to-End Test")
    print("=" * 65)
    results = []

    # Pre-checks
    print("\n--- Pre-checks ---")
    results.append(test_api_key())
    results.append(test_machamp_override())
    results.append(test_variant_display())

    if not results[0]:
        print("\nAPI key not loaded — skipping API tests.")
        sys.exit(1)

    # API pricing tests
    print("\n--- Base Set (Shadowless map) ---")
    results.append(test_pricing("base1-4", True, "Rare Holo",
                                "Charizard 1st Ed Holo"))
    results.append(test_pricing("base1-4", False, "Rare Holo",
                                "Charizard Unlimited Holo"))
    results.append(test_pricing("base1-20", True, "Rare",
                                "Electabuzz 1st Ed Non-Holo"))

    print("\n--- Item 2: Machamp (override) ---")
    results.append(test_pricing("base1-8", True, "Rare Holo",
                                "Machamp 1st Ed Holo"))

    print("\n--- Item 1: Jungle/Fossil (base2/base3) ---")
    results.append(test_pricing("base2-3", True, "Rare Holo",
                                "Flareon 1st Ed Holo (Jungle)"))
    results.append(test_pricing("base2-3", False, "Rare Holo",
                                "Flareon Unlimited (Jungle)"))
    results.append(test_pricing("base3-1", True, "Rare Holo",
                                "Aerodactyl 1st Ed (Fossil)"))

    print("\n--- Team Rocket (base5) ---")
    results.append(test_pricing("base5-4", True, "Rare Holo",
                                "Dark Charizard 1st Ed Holo"))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 65}")
    print(f"Results: {passed}/{total} passed")
    if passed == total:
        print("All tests passed.")
    else:
        print(f"{total - passed} test(s) failed.")
    print(f"{'=' * 65}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()

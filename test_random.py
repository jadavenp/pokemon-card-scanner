"""
test_random.py
Picks random cards from Ref Images, runs the full scanner pipeline,
then cleans up. No files accumulate between runs.

Uses symlinks (not copies) to avoid duplicating image data.
Cleans up before AND after each run to prevent accumulation.

Usage:
    python3 test_random.py              # 4 random cards (default)
    python3 test_random.py -n 8         # 8 random cards
    python3 test_random.py --verbose    # show debug output
    python3 test_random.py --ui         # open results in browser
    python3 test_random.py --no-hash    # skip hash matching
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

from config import DATA_DIR, IMAGE_DIR, SET_NAMES
from database import load_database
from stamp import load_stamp_template
from scanner import process_image, print_summary_table

try:
    from image_match import load_hash_database
    IMAGE_MATCH_AVAILABLE = True
except ImportError:
    IMAGE_MATCH_AVAILABLE = False

REF_IMAGES_DIR = DATA_DIR / "Ref Images"
# Prefix for temp symlinks so we can identify and clean them up
TEST_PREFIX = "_test_"


def cleanup_test_files():
    """Remove any leftover test symlinks/files from previous runs."""
    count = 0
    for f in Path(IMAGE_DIR).iterdir():
        if f.name.startswith(TEST_PREFIX):
            f.unlink()
            count += 1
    return count


def pick_random_cards(n):
    """Pick n random card images from Ref Images."""
    if not REF_IMAGES_DIR.exists():
        print(f"Error: {REF_IMAGES_DIR} not found.")
        sys.exit(1)

    all_images = [f for f in REF_IMAGES_DIR.iterdir()
                  if f.suffix.lower() == ".png"]

    if len(all_images) < n:
        print(f"Warning: Only {len(all_images)} images available, using all of them.")
        n = len(all_images)

    selected = random.sample(all_images, n)
    return selected


def link_test_cards(selected):
    """Create symlinks in data/ for the selected test cards.
    Returns list of symlink paths."""
    links = []
    for img_path in selected:
        # Symlink name: _test_base1-58.png
        link_name = f"{TEST_PREFIX}{img_path.name}"
        link_path = Path(IMAGE_DIR) / link_name
        if link_path.exists():
            link_path.unlink()
        os.symlink(img_path.resolve(), link_path)
        links.append(link_path)
    return links


def main():
    parser = argparse.ArgumentParser(description="Test scanner on random cards")
    parser.add_argument("-n", type=int, default=4,
                        help="Number of random cards to test (default: 4)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show debug output")
    parser.add_argument("--no-hash", action="store_true",
                        help="Disable image hash matching")
    parser.add_argument("--ui", action="store_true",
                        help="Generate HTML results and open in browser")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible tests")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    overall_start = time.time()

    # Clean up any leftover test files from previous runs
    cleaned = cleanup_test_files()
    if cleaned > 0:
        print(f"Cleaned up {cleaned} leftover test files.")

    # Load resources
    print("Loading resources...")
    index, card_count = load_database()
    print(f"  {card_count} cards in database")

    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("  OCR ready")

    stamp_template = load_stamp_template()
    print(f"  Stamp template: {'loaded' if stamp_template is not None else 'not found'}")

    hash_db = None
    if IMAGE_MATCH_AVAILABLE and not args.no_hash:
        hash_db = load_hash_database()
        if hash_db:
            print(f"  Hash database: {len(hash_db)} cards")

    # Pick random cards
    print(f"\nPicking {args.n} random cards from {len(list(REF_IMAGES_DIR.glob('*.png')))} available...")
    selected = pick_random_cards(args.n)
    for img in selected:
        card_id = img.stem
        print(f"  → {card_id}")

    # Create symlinks
    test_files = link_test_cards(selected)

    # Run scanner on each
    print(f"\nScanning {len(test_files)} cards...")
    print("=" * 70)

    results = []
    for i, img_path in enumerate(test_files):
        print(f"\n[{i+1}/{len(test_files)}] {img_path.name.replace(TEST_PREFIX, '')}...")
        t0 = time.time()
        r = process_image(img_path, reader, index, stamp_template,
                          hash_db=hash_db, verbose=args.verbose)
        elapsed = time.time() - t0
        r["time"] = elapsed

        # Clean up the display filename (strip test prefix)
        r["file"] = r["file"].replace(TEST_PREFIX, "")

        if r.get("error"):
            print(f"  ERROR: {r['error']}")
        else:
            edition = "1st Ed" if r["stamp_1st"] else "Unltd"
            id_tag = f" [{r.get('id_method', '?')}]"
            print(f"  Name: {r['name']} ({r['name_conf']}%) | "
                  f"Num: {r['number']}/{r['total']} ({r['num_conf']}%) | "
                  f"Stamp: {edition} ({r['stamp_conf']}%){id_tag}")
            if r["card_id"]:
                hash_info = ""
                if r.get("hash_distance") is not None:
                    hash_info = f" | Hash dist: {r['hash_distance']}"
                print(f"  Match: {r['card_id']} -> {r['set_name']} | "
                      f"Rarity: {r['rarity']} | Price: {r['price']}{hash_info}")

                # Verify: does the hash match the actual card?
                actual_id = img_path.stem.replace(TEST_PREFIX, "")
                if r["card_id"] == actual_id:
                    print(f"  ✓ CORRECT (matched actual card ID)")
                else:
                    print(f"  ✗ MISMATCH — actual: {actual_id}, got: {r['card_id']}")
            else:
                print(f"  No database match")
        print(f"  ({elapsed:.1f}s)")

        results.append(r)

    # Summary
    total_time = time.time() - overall_start

    print(f"\n\n{'=' * 100}")
    print("RANDOM TEST RESULTS")
    print(f"{'=' * 100}\n")

    table, stats = print_summary_table(results)
    print(table)

    # Accuracy check
    correct = 0
    incorrect = 0
    for i, r in enumerate(results):
        actual_id = selected[i].stem
        if r.get("card_id") == actual_id:
            correct += 1
        elif r.get("card_id"):
            incorrect += 1

    identified = stats["identified"]
    failed = len(results) - identified

    print(f"\n{'─' * 50}")
    print(f"  Cards tested:     {len(results)}")
    print(f"  Identified:       {identified}")
    print(f"  Correct ID:       {correct}")
    print(f"  Wrong ID:         {incorrect}")
    print(f"  Failed:           {failed}")
    print(f"  Accuracy:         {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
    print(f"  Total value:      ${stats['total_price']:.2f}")
    if hash_db:
        print(f"  Hash confirmed:   {stats['hash_confirmed']}")
        print(f"  Hash-only ID:     {stats['hash_only']}")
    print(f"  Total time:       {total_time:.1f}s ({total_time/max(len(results),1):.1f}s avg)")
    print(f"{'─' * 50}")

    # Generate HTML if requested
    if args.ui:
        try:
            from scanner_ui import generate_html
            import webbrowser
            html = generate_html(results, total_time)
            output_path = Path(DATA_DIR) / "scan_results.html"
            with open(output_path, "w") as f:
                f.write(html)
            webbrowser.open(f"file://{output_path.resolve()}")
            print(f"\nResults opened in browser: {output_path}")
        except ImportError:
            print("\nWarning: scanner_ui module not found — --ui flag ignored.")
            print("(scanner_ui.py was removed during cleanup; HTML UI is now served by server.py)")

    # Clean up test symlinks
    cleanup_test_files()
    print(f"\nCleaned up test files.")


if __name__ == "__main__":
    main()

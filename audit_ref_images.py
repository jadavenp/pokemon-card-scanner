"""
audit_ref_images.py
Scans the data/ref_images/ folder and produces a report of all image filenames,
organized by set, with gap detection.

Output: data/ref_images_audit.txt

Usage:
    python3 audit_ref_images.py
    python3 audit_ref_images.py --json     # Also output data/ref_images_audit.json
    python3 audit_ref_images.py --compare   # Compare against card_index.json to find gaps
"""

import os
import sys
import re
import json
import argparse
from collections import defaultdict

# ============================================
# CONFIG
# ============================================
DATA_DIR = "data"
REF_IMAGES_DIR = os.path.join(DATA_DIR, "Ref Images")
CARD_INDEX_FILE = os.path.join(DATA_DIR, "card_index.json")
OUTPUT_TXT = os.path.join(DATA_DIR, "ref_images_audit.txt")
OUTPUT_JSON = os.path.join(DATA_DIR, "ref_images_audit.json")

# Image filename pattern: {set_id}-{number}.png
# Examples: base1-1.png, swsh12pt5-160.png, sv3pt5-186.png
IMAGE_PATTERN = re.compile(r"^(.+)-(\d+)\.png$")


def scan_images(images_dir):
    """Scan the ref images folder and parse filenames into set/number pairs."""
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} not found.")
        print("Make sure you're running this from the card_scanner directory.")
        sys.exit(1)

    files = os.listdir(images_dir)
    image_files = sorted([f for f in files if f.lower().endswith(".png")])
    non_image_files = [f for f in files if not f.lower().endswith(".png")]

    parsed = []      # (set_id, number, filename)
    unparsed = []    # Filenames that don't match the pattern

    for fname in image_files:
        match = IMAGE_PATTERN.match(fname)
        if match:
            set_id = match.group(1)
            number = int(match.group(2))
            parsed.append((set_id, number, fname))
        else:
            unparsed.append(fname)

    return image_files, parsed, unparsed, non_image_files


def analyze_by_set(parsed):
    """Group images by set and find gaps in numbering."""
    sets = defaultdict(list)
    for set_id, number, fname in parsed:
        sets[set_id].append((number, fname))

    # Sort each set's cards by number
    for set_id in sets:
        sets[set_id].sort(key=lambda x: x[0])

    return dict(sets)


def find_gaps(numbers):
    """Given a list of card numbers, find gaps in the sequence."""
    if not numbers:
        return []
    min_n = min(numbers)
    max_n = max(numbers)
    full_range = set(range(min_n, max_n + 1))
    present = set(numbers)
    missing = sorted(full_range - present)
    return missing


def compare_with_index(sets_data, card_index_file):
    """Compare ref images against card_index.json to find what's missing."""
    if not os.path.exists(card_index_file):
        return None

    with open(card_index_file, "r", encoding="utf-8") as f:
        card_index = json.load(f)

    index_by_id = card_index.get("index", {}).get("by_id", {})
    sets_meta = card_index.get("sets", {})

    # Build set of image card IDs we have
    image_ids = set()
    for set_id, cards in sets_data.items():
        for number, fname in cards:
            card_id = f"{set_id}-{number}"
            image_ids.add(card_id)

    # All card IDs in the index
    index_ids = set(index_by_id.keys())

    # Compare
    in_images_not_index = sorted(image_ids - index_ids)
    in_index_not_images = sorted(index_ids - image_ids)
    in_both = image_ids & index_ids

    return {
        "index_total": len(index_ids),
        "images_total": len(image_ids),
        "matched": len(in_both),
        "images_only": in_images_not_index,
        "index_only": in_index_not_images,
        "sets_meta": sets_meta,
    }


def write_report(image_files, parsed, unparsed, non_image_files, sets_data, comparison):
    """Write the human-readable audit report."""
    lines = []
    lines.append("=" * 70)
    lines.append("POKEMON TCG REFERENCE IMAGES — AUDIT REPORT")
    lines.append(f"Generated: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source: {REF_IMAGES_DIR}")
    lines.append("=" * 70)
    lines.append("")

    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"  Total files:         {len(image_files) + len(non_image_files)}")
    lines.append(f"  PNG images:          {len(image_files)}")
    lines.append(f"  Parsed (set-num):    {len(parsed)}")
    lines.append(f"  Unparsed filenames:  {len(unparsed)}")
    lines.append(f"  Non-image files:     {len(non_image_files)}")
    lines.append(f"  Unique sets:         {len(sets_data)}")
    lines.append("")

    # Comparison with card index
    if comparison:
        lines.append("COMPARISON WITH CARD INDEX")
        lines.append("-" * 40)
        lines.append(f"  Cards in index:        {comparison['index_total']}")
        lines.append(f"  Images on disk:        {comparison['images_total']}")
        lines.append(f"  Matched (both):        {comparison['matched']}")
        lines.append(f"  Images w/o index:      {len(comparison['images_only'])}")
        lines.append(f"  Index w/o images:      {len(comparison['index_only'])}")
        coverage = (comparison['matched'] / comparison['index_total'] * 100) if comparison['index_total'] > 0 else 0
        lines.append(f"  Coverage:              {coverage:.1f}%")
        lines.append("")

    # Per-set summary table
    lines.append("PER-SET SUMMARY")
    lines.append("-" * 70)
    lines.append(f"  {'Set ID':<20s}  {'Count':>6s}  {'Range':>12s}  {'Gaps':>5s}")
    lines.append(f"  {'------':<20s}  {'-----':>6s}  {'-----':>12s}  {'----':>5s}")

    for set_id in sorted(sets_data.keys()):
        cards = sets_data[set_id]
        numbers = [n for n, _ in cards]
        gaps = find_gaps(numbers)
        min_n = min(numbers)
        max_n = max(numbers)
        range_str = f"{min_n}-{max_n}"
        lines.append(f"  {set_id:<20s}  {len(cards):>6d}  {range_str:>12s}  {len(gaps):>5d}")

    lines.append("")

    # Per-set detail with gaps
    lines.append("PER-SET DETAIL (sets with gaps)")
    lines.append("-" * 70)
    for set_id in sorted(sets_data.keys()):
        cards = sets_data[set_id]
        numbers = [n for n, _ in cards]
        gaps = find_gaps(numbers)
        if gaps:
            lines.append(f"\n  {set_id} ({len(cards)} images, {len(gaps)} gaps)")
            # Show gap ranges compactly
            gap_ranges = []
            start = gaps[0]
            end = gaps[0]
            for g in gaps[1:]:
                if g == end + 1:
                    end = g
                else:
                    if start == end:
                        gap_ranges.append(str(start))
                    else:
                        gap_ranges.append(f"{start}-{end}")
                    start = g
                    end = g
            if start == end:
                gap_ranges.append(str(start))
            else:
                gap_ranges.append(f"{start}-{end}")
            lines.append(f"    Missing: {', '.join(gap_ranges)}")

    lines.append("")

    # Unparsed filenames
    if unparsed:
        lines.append("UNPARSED FILENAMES (don't match set-number.png pattern)")
        lines.append("-" * 40)
        for f in unparsed:
            lines.append(f"  {f}")
        lines.append("")

    # Non-image files
    if non_image_files:
        lines.append("NON-IMAGE FILES")
        lines.append("-" * 40)
        for f in non_image_files:
            lines.append(f"  {f}")
        lines.append("")

    # Index-only cards (top missing sets if comparing)
    if comparison and comparison["index_only"]:
        lines.append("TOP SETS MISSING IMAGES (from card index)")
        lines.append("-" * 40)
        missing_by_set = defaultdict(list)
        for card_id in comparison["index_only"]:
            # Card ID format: setid-number
            parts = card_id.rsplit("-", 1)
            if len(parts) == 2:
                missing_by_set[parts[0]].append(parts[1])

        # Sort by count descending
        for set_id, nums in sorted(missing_by_set.items(), key=lambda x: -len(x[1]))[:20]:
            set_name = comparison["sets_meta"].get(set_id, {}).get("name", "")
            lines.append(f"  {set_id:<20s}  {len(nums):>5d} missing  {set_name}")
        if len(missing_by_set) > 20:
            lines.append(f"  ... and {len(missing_by_set) - 20} more sets")
        lines.append("")

    # Complete file list
    lines.append("COMPLETE FILE LIST")
    lines.append("-" * 70)
    for set_id in sorted(sets_data.keys()):
        cards = sets_data[set_id]
        filenames = [fname for _, fname in cards]
        lines.append(f"\n  [{set_id}] ({len(cards)} files)")
        # Print in columns
        col_width = 30
        cols = 3
        for i in range(0, len(filenames), cols):
            row = filenames[i : i + cols]
            lines.append("    " + "".join(f"{f:<{col_width}}" for f in row))

    return "\n".join(lines)


def write_json_report(image_files, parsed, unparsed, sets_data, comparison):
    """Write the machine-readable JSON audit."""
    report = {
        "total_images": len(image_files),
        "parsed_count": len(parsed),
        "unparsed": unparsed,
        "sets": {},
    }

    for set_id in sorted(sets_data.keys()):
        cards = sets_data[set_id]
        numbers = [n for n, _ in cards]
        filenames = [fname for _, fname in cards]
        gaps = find_gaps(numbers)
        report["sets"][set_id] = {
            "count": len(cards),
            "min": min(numbers),
            "max": max(numbers),
            "gaps": gaps,
            "filenames": filenames,
        }

    if comparison:
        report["comparison"] = {
            "index_total": comparison["index_total"],
            "images_total": comparison["images_total"],
            "matched": comparison["matched"],
            "images_only_count": len(comparison["images_only"]),
            "index_only_count": len(comparison["index_only"]),
        }

    return report


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description="Audit Pokemon TCG reference images")
    parser.add_argument("--json", action="store_true", help="Also output JSON audit file")
    parser.add_argument("--compare", action="store_true", help="Compare against card_index.json")
    args = parser.parse_args()

    print("=" * 60)
    print("Pokemon TCG Reference Images — Audit")
    print("=" * 60)
    print()

    # Scan images
    print(f"Scanning {REF_IMAGES_DIR}...")
    image_files, parsed, unparsed, non_image_files = scan_images(REF_IMAGES_DIR)
    print(f"  Found {len(image_files)} PNG files")
    print(f"  Parsed: {len(parsed)}, Unparsed: {len(unparsed)}")

    # Organize by set
    print("\nAnalyzing by set...")
    sets_data = analyze_by_set(parsed)
    print(f"  {len(sets_data)} unique sets")

    # Optional: compare with card index
    comparison = None
    if args.compare:
        print(f"\nComparing with {CARD_INDEX_FILE}...")
        comparison = compare_with_index(sets_data, CARD_INDEX_FILE)
        if comparison:
            print(f"  Index: {comparison['index_total']} cards")
            print(f"  Images: {comparison['images_total']} files")
            print(f"  Matched: {comparison['matched']}")
            print(f"  Coverage: {comparison['matched'] / comparison['index_total'] * 100:.1f}%")
        else:
            print(f"  {CARD_INDEX_FILE} not found — run build_card_index.py first")

    # Write text report
    print(f"\nWriting report to {OUTPUT_TXT}...")
    report_text = write_report(image_files, parsed, unparsed, non_image_files, sets_data, comparison)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Done: {os.path.getsize(OUTPUT_TXT) / 1024:.0f} KB")

    # Optional JSON
    if args.json:
        print(f"\nWriting JSON to {OUTPUT_JSON}...")
        json_report = write_json_report(image_files, parsed, unparsed, sets_data, comparison)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        print(f"  Done: {os.path.getsize(OUTPUT_JSON) / 1024:.0f} KB")

    # Quick console summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    print(f"  Images:  {len(parsed)}")
    print(f"  Sets:    {len(sets_data)}")

    # Top 10 sets by image count
    top_sets = sorted(sets_data.items(), key=lambda x: -len(x[1]))[:10]
    print(f"\n  Top 10 sets by image count:")
    for set_id, cards in top_sets:
        print(f"    {set_id:<20s}  {len(cards):>5d} images")

    print(f"\n  Full report: {OUTPUT_TXT}")
    if args.json:
        print(f"  JSON report: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

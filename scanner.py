"""
scanner.py
Pokemon card scanner — main entry point.
Batch-processes card images in data/, identifies via OCR + local DB,
detects 1st Edition stamps, fetches live pricing via JustTCG API.
Optionally uses perceptual image hashing for confirmation/fallback ID.

Requires:
    - data/tcgplayer_id_map.json (built by build_tcgplayer_map.py)
    - .env with JUSTTCG_API_KEY

Usage:
    cd ~/card-scanner && source venv/bin/activate
    python3 scanner.py                  # normal output
    python3 scanner.py --verbose        # extra debug info
    python3 scanner.py --quiet          # summary table only
    python3 scanner.py --no-hash        # disable image hash matching
"""

import argparse
import cv2
import easyocr
import re
import sys
import time
import warnings
from pathlib import Path
from card_detect import detect_and_crop_card

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from prettytable import PrettyTable
except ImportError:
    print("Missing dependency: pip install prettytable")
    sys.exit(1)

from config import IMAGE_DIR, SET_NAMES, VARIANT_DISPLAY, CONFIDENCE_NAME_WEIGHT, CONFIDENCE_NUMBER_WEIGHT, CONFIDENCE_HASH_WEIGHT, CONFIDENCE_STAMP_WEIGHT, KNOWN_SET_TOTALS
from database import load_database, lookup_card, lookup_by_name, get_set_id
from ocr import detect_card_type, extract_name, extract_number
from pricing_justtcg import fetch_live_pricing
from stamp import load_stamp_template, check_stamp

# Image matching is optional — gracefully degrade if not available
try:
    from image_match import load_hash_database, match_card_image
    IMAGE_MATCH_AVAILABLE = True
except ImportError:
    IMAGE_MATCH_AVAILABLE = False




def compute_identification_confidence(result):
    """
    Compute a composite identification confidence score (0-100%)
    based on weighted combination of all identification signals.

    Signals:
      - Name OCR confidence (0-100)
      - Number OCR confidence (0-100)
      - Hash match confidence (0 = no match, 100 = perfect match)
      - Stamp detection confidence (informational, low weight)

    Returns a float 0-100 representing overall ID confidence.
    """
    name_conf = result.get("name_conf", 0.0)
    num_conf = result.get("num_conf", 0.0)
    stamp_conf = result.get("stamp_conf", 0.0)

    # Convert hash distance to confidence (0 distance = 100%, 15+ = 0%)
    hash_dist = result.get("hash_distance")
    if hash_dist is not None:
        hash_conf = max(0, min(100, (15 - hash_dist) / 15 * 100))
    else:
        hash_conf = 0.0

    # Weighted combination
    # If a signal is missing (0), its weight redistributes to available signals
    weights = {}
    if name_conf > 0:
        weights["name"] = CONFIDENCE_NAME_WEIGHT
    if num_conf > 0:
        weights["number"] = CONFIDENCE_NUMBER_WEIGHT
    if hash_conf > 0:
        weights["hash"] = CONFIDENCE_HASH_WEIGHT
    if stamp_conf > 0:
        weights["stamp"] = CONFIDENCE_STAMP_WEIGHT

    if not weights:
        return 0.0

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    composite = 0.0
    if "name" in weights:
        composite += (weights["name"] / total_weight) * name_conf
    if "number" in weights:
        composite += (weights["number"] / total_weight) * num_conf
    if "hash" in weights:
        composite += (weights["hash"] / total_weight) * hash_conf
    if "stamp" in weights:
        composite += (weights["stamp"] / total_weight) * stamp_conf

    return round(composite, 1)


def validate_number_against_sets(number, total, candidate_sets=None):
    """
    Cross-check an extracted set total against known set sizes.
    Returns list of matching set IDs, or empty list if no match.
    """
    if not total:
        return []
    matching = KNOWN_SET_TOTALS.get(total, [])
    if candidate_sets and matching:
        # Intersect with candidate sets
        filtered = [s for s in matching if s in candidate_sets]
        return filtered if filtered else matching
    return matching


def process_image(img_path, reader, index, stamp_template, hash_db=None,
                  verbose=False):
    """
    Process one card image. Returns a dict with all extracted fields.

    Identification strategy:
      1. OCR extracts name + number → database lookup
      2. If hash_db is loaded, image hash provides confirmation or fallback
      3. If OCR fails but hash matches confidently → use hash result
      4. If both succeed but disagree → flag in id_method
    """
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        return {"file": img_path.name, "error": "Could not load image"}

    # Run card detection + perspective correction on all inputs
    img, card_detected = detect_and_crop_card(img)
    if card_detected:
        # Overwrite temp file so downstream steps use corrected image
        cv2.imwrite(str(img_path), img)

    result = {
        "file": img_path.name,
        "name": None, "name_conf": 0.0,
        "number": None, "total": None, "num_conf": 0.0,
        "card_type": None,
        "stamp_1st": False, "stamp_conf": 0.0,
        "set_name": None, "card_id": None, "rarity": None,
        "price": "N/A", "price_variant": "", "price_src": "",
        "hash_match": None, "hash_distance": None, "hash_confident": False,
        "id_method": None,  # "ocr", "hash", "ocr+hash", "ocr (hash disagree)"
        "error": None,
    }

    # ── OCR Pipeline ──
    ocr_results = reader.readtext(img)
    result["card_type"] = detect_card_type(ocr_results)
    name, name_conf = extract_name(ocr_results)
    result["name"] = name
    result["name_conf"] = name_conf

    num, total, num_conf = extract_number(img, reader, card_type=result['card_type'])
    result["number"] = num
    result["total"] = total
    result["num_conf"] = num_conf

    # Fallback: search Pass 1 results for number pattern
    if not num:
        number_pattern = re.compile(r'(\d{1,3})\s*/\s*(\d{1,3})')
        for (bbox, text, conf) in ocr_results:
            match = number_pattern.search(text)
            if match:
                result["number"] = match.group(1)
                result["total"] = match.group(2)
                result["num_conf"] = round(conf * 100, 1)
                break

    # ── Image Hash Matching ──
    hash_result = None
    if hash_db is not None:
        hash_result = match_card_image(img, hash_db)
        if hash_result["match"]:
            result["hash_match"] = hash_result["card_id"]
            result["hash_distance"] = hash_result["distance"]
            result["hash_confident"] = hash_result["confident"]

            if verbose and hash_result["candidates"]:
                top = hash_result["candidates"][0]
                print(f"    [hash] best={top['card_id']} dist={top['distance']} "
                      f"(p={top['distances']['phash']} d={top['distances']['dhash']} "
                      f"w={top['distances']['whash']})"
                      f"{' CONFIDENT' if top['confident'] else ''}")

    # ── Database Lookup + Identification (confidence-based) ──
    ocr_matched = False
    id_confidence = 0.0

    # Strategy 1: Name + Number → strongest match
    if result["name"] and result["number"]:
        matches = lookup_card(index, result["name"], result["number"])
        if matches:
            card = matches[0]
            card_id = card.get("id", "")
            set_id = get_set_id(card)
            result["card_id"] = card_id
            result["set_name"] = SET_NAMES.get(set_id, set_id)
            result["rarity"] = card.get("rarity", "?")
            ocr_matched = True

            if hash_result and hash_result["match"]:
                if hash_result["card_id"] == card_id:
                    result["id_method"] = "ocr+hash"
                else:
                    result["id_method"] = "ocr (hash disagree)"
            else:
                result["id_method"] = "ocr"

    # Strategy 2: Hash fallback if OCR name+number failed
    if not ocr_matched and hash_result and hash_result["match"] and hash_result["confident"]:
        result["card_id"] = hash_result["card_id"]
        result["name"] = hash_result["name"]
        set_id = hash_result["card_id"].rsplit("-", 1)[0] if "-" in hash_result["card_id"] else ""
        result["set_name"] = SET_NAMES.get(set_id, set_id)
        result["id_method"] = "hash"
        card_data = index.get("by_id", {}).get(hash_result["card_id"])
        if card_data:
            result["rarity"] = card_data.get("rarity", "?")
            result["number"] = card_data.get("number", "")
        ocr_matched = True

    # Strategy 3: Name-only fallback (number OCR failed but name is good)
    if not ocr_matched and result["name"] and result["name_conf"] > 30:
        # Use set total hint from number extraction if available
        set_hints = validate_number_against_sets(result["number"], result["total"])
        set_hint = set_hints[0] if len(set_hints) == 1 else None

        name_matches = lookup_by_name(index, result["name"], set_hint=set_hint)
        if name_matches:
            if len(name_matches) == 1:
                # Unambiguous: only one card with this name
                card = name_matches[0]
                result["card_id"] = card.get("id", "")
                set_id = get_set_id(card)
                result["set_name"] = SET_NAMES.get(set_id, set_id)
                result["rarity"] = card.get("rarity", "?")
                result["number"] = card.get("number", result["number"])
                result["id_method"] = "name_only"
                ocr_matched = True
            else:
                # Ambiguous: multiple cards share this name — pick best candidate
                # Prefer vintage sets, or use hash hint if available
                card = name_matches[0]  # Already sorted vintage-first
                result["card_id"] = card.get("id", "")
                set_id = get_set_id(card)
                result["set_name"] = SET_NAMES.get(set_id, set_id)
                result["rarity"] = card.get("rarity", "?")
                result["number"] = card.get("number", result["number"])
                result["id_method"] = f"name_only ({len(name_matches)} candidates)"
                ocr_matched = True

    # ── Compute identification confidence ──
    id_confidence = compute_identification_confidence(result)
    result["id_confidence"] = id_confidence

    # ── Pricing (always attempt on best candidate, show confidence alongside) ──
    if ocr_matched and result["card_id"]:
        from pricing_cache import fetch_cached_pricing
        pricing = fetch_cached_pricing(
            result["card_id"],
            is_1st_edition=result["stamp_1st"],
            rarity=result.get("rarity", "?"),
            use_cache=True,  # Default to batch mode; server.py can override
        )
        result["price"] = pricing["price"]
        result["price_variant"] = pricing["variant"]
        result["price_src"] = pricing["source"]

    if not ocr_matched:
        result["error"] = "Identification failed"
        if result["name"]:
            result["error"] = f"Name '{result['name']}' not found in database"
        if hash_result and hash_result["match"] and not hash_result["confident"]:
            result["error"] = f"OCR incomplete (hash tentative: {hash_result['card_id']})"

    return result


def find_card_images():
    """Find all card images in the data directory."""
    extensions = ("*.png", "*.jpg", "*.jpeg")
    exclude_files = {"stamp_template.png"}
    image_files = []
    for ext in extensions:
        for f in Path(IMAGE_DIR).glob(ext):
            if f.name not in exclude_files:
                image_files.append(f)
    return sorted(image_files)


def print_summary_table(results):
    """Print the batch results summary table."""
    table = PrettyTable()
    table.field_names = [
        "File", "Card (Set)", "Num", "Ed",
        "Price", "Variant", "ID", "Name%", "Num%", "Stamp%", "Time"
    ]
    table.align["File"] = "l"
    table.align["Card (Set)"] = "l"
    table.align["Price"] = "r"
    table.align["Variant"] = "l"
    table.align["ID"] = "l"
    table.align["Time"] = "r"
    table.max_width["File"] = 28
    table.max_width["Card (Set)"] = 28
    table.max_width["Variant"] = 12
    table.max_width["ID"] = 12
    table.min_width["File"] = 28
    table.hrules = 1

    total_price = 0.0
    identified = 0
    first_eds = 0
    api_calls = 0
    hash_confirmed = 0
    hash_only = 0

    for r in results:
        if r.get("error"):
            table.add_row([
                r["file"], f"ERROR: {r['error']}", "-", "-",
                "-", "-", "-", "-", "-", "-", f"{r.get('time', 0):.1f}s"
            ])
            continue

        name = r["name"] or "?"
        set_name = r["set_name"] or "?"
        num_str = f"{r['number']}/{r['total']}" if r["number"] else "?"
        edition = "1st" if r["stamp_1st"] else "Unl"
        price = r["price"]
        variant = r.get("price_variant", "")
        elapsed = r.get("time", 0)
        id_method = r.get("id_method", "?")

        card_display = f"{name} ({set_name})"
        variant_short = VARIANT_DISPLAY.get(variant, variant)

        id_display = {
            "ocr": "OCR",
            "hash": "Hash",
            "ocr+hash": "OCR+Hash",
            "ocr (hash disagree)": "OCR(!H)",
        }.get(id_method, id_method or "?")

        identified += 1
        if r["stamp_1st"]:
            first_eds += 1
        if price.startswith("$"):
            try:
                total_price += float(price[1:])
            except ValueError:
                pass
        if r.get("price_src"):
            api_calls += 1
        if id_method == "ocr+hash":
            hash_confirmed += 1
        elif id_method == "hash":
            hash_only += 1

        table.add_row([
            r["file"], card_display, num_str, edition,
            price, variant_short, id_display,
            f"{r['name_conf']}%", f"{r['num_conf']}%",
            f"{r['stamp_conf']}%", f"{elapsed:.1f}s",
        ])

    stats = {
        "identified": identified,
        "first_eds": first_eds,
        "total_price": total_price,
        "api_calls": api_calls,
        "hash_confirmed": hash_confirmed,
        "hash_only": hash_only,
    }
    return table, stats


def main():
    parser = argparse.ArgumentParser(description="Pokemon Card Scanner")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show extra debug output")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Show summary table only")
    parser.add_argument("--no-hash", action="store_true",
                        help="Disable image hash matching")
    args = parser.parse_args()

    overall_start = time.time()

    # Load resources
    if not args.quiet:
        print("Loading local card database...")
    index, card_count = load_database()
    if not args.quiet:
        print(f"  {card_count} cards loaded")

    if not args.quiet:
        print("Loading OCR model...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    if not args.quiet:
        print("  OCR ready")

    stamp_template = load_stamp_template()
    if not args.quiet:
        if stamp_template is not None:
            print(f"  1st Edition template loaded ({stamp_template.shape[1]}x{stamp_template.shape[0]})")
        else:
            print("  1st Edition template not found — stamp detection disabled")

    # Load hash database (optional)
    hash_db = None
    if IMAGE_MATCH_AVAILABLE and not args.no_hash:
        hash_db = load_hash_database()
        if not args.quiet:
            if hash_db:
                print(f"  Hash database loaded ({len(hash_db)} cards)")
            else:
                print("  Hash database not found — run build_hash_db.py to enable")
    elif not IMAGE_MATCH_AVAILABLE and not args.quiet:
        print("  Image matching unavailable (pip install imagehash)")

    # Find images
    image_files = find_card_images()
    if not image_files:
        print(f"\nNo images found in {IMAGE_DIR}")
        sys.exit(1)

    if not args.quiet:
        print(f"\nFound {len(image_files)} image(s) in {IMAGE_DIR}")
        print("=" * 70)

    # Process
    results = []
    for i, img_path in enumerate(image_files):
        if not args.quiet:
            print(f"\n[{i+1}/{len(image_files)}] Processing {img_path.name}...")
        t0 = time.time()
        r = process_image(img_path, reader, index, stamp_template,
                          hash_db=hash_db, verbose=args.verbose)
        elapsed = time.time() - t0
        r["time"] = elapsed

        if not args.quiet:
            if r.get("error"):
                print(f"  ERROR: {r['error']}")
            else:
                edition = "1st Ed" if r["stamp_1st"] else "Unltd"
                id_tag = f" [{r.get('id_method', '?')}]"
                print(f"  Name: {r['name']} ({r['name_conf']}%) | "
                      f"Num: {r['number']}/{r['total']} ({r['num_conf']}%) | "
                      f"Stamp: {edition} ({r['stamp_conf']}%){id_tag}")
                if r["card_id"]:
                    variant_info = f" [{r['price_variant']}]" if r.get("price_variant") else ""
                    src_info = f" via {r['price_src']}" if r.get("price_src") else ""
                    hash_info = ""
                    if r.get("hash_distance") is not None:
                        hash_info = f" | Hash dist: {r['hash_distance']}"
                    print(f"  Match: {r['card_id']} -> {r['set_name']} | "
                          f"Rarity: {r['rarity']} | Price: {r['price']}"
                          f"{variant_info}{src_info}{hash_info}")
                else:
                    print(f"  No database match")
            print(f"  ({elapsed:.1f}s)")

        if args.verbose and not r.get("error"):
            print(f"    [debug] card_type={r['card_type']}, "
                  f"stamp_1st={r['stamp_1st']}, rarity={r['rarity']}, "
                  f"id_method={r.get('id_method')}")

        results.append(r)

    # Summary
    print(f"\n\n{'=' * 100}")
    print("BATCH RESULTS SUMMARY")
    print(f"{'=' * 100}\n")

    table, stats = print_summary_table(results)
    print(table)

    total_time = time.time() - overall_start
    identified = stats["identified"]
    first_eds = stats["first_eds"]
    total_price = stats["total_price"]
    api_calls = stats["api_calls"]
    failed = len(results) - identified

    print(f"\n{'─' * 50}")
    print(f"  Cards scanned:    {len(results)}")
    print(f"  Identified:       {identified}")
    print(f"  Failed:           {failed}")
    print(f"  1st Edition:      {first_eds}")
    print(f"  Unlimited:        {identified - first_eds}")
    print(f"  Total value:      ${total_price:.2f}")
    print(f"  API calls:        JustTCG: {api_calls} / 100 daily")
    if hash_db:
        print(f"  Hash confirmed:   {stats['hash_confirmed']}")
        print(f"  Hash-only ID:     {stats['hash_only']}")
    print(f"  Total time:       {total_time:.1f}s ({total_time/max(len(results),1):.1f}s avg)")
    if first_eds > 0:
        print(f"\n  * 1st Ed cards priced via JustTCG 1st Edition printing filter")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    main()

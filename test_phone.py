"""
test_phone.py
Phase 1 — Measure perceptual hash robustness on phone camera photos.

PURPOSE:
    Answers the gating questions before any pipeline changes:
    1. What's the hash distance distribution for phone photos of known cards?
    2. Does DISTANCE_CONFIDENT (8) still separate true matches from false positives?
    3. What % of phone scans would need OCR fallback?
    4. How stable is card_detect.py's perspective correction on phone input?
    5. Does hash distance degrade for sleeved vs. bare cards?

WORKFLOW:
    1. Enable "Test Mode" in the mobile scanner UI (scanner_mobile.html).
    2. Set condition to Bare or Sleeve before each scan.
    3. Scan 20-40 cards from the shop counter under real lighting.
       → Server auto-saves each upload as capture_NNN.jpg and logs
         scan results to data/test_phone/manifest.json.
    4. Run this script to re-analyze all captures:

           cd ~/card-scanner && source venv/bin/activate
           python3 test_phone.py                     # default analysis
           python3 test_phone.py --save-crops        # save intermediate images
           python3 test_phone.py --csv               # export to CSV
           python3 test_phone.py --verbose            # per-image progress
           python3 test_phone.py --list-ids sv        # browse hash DB

OUTPUT:
    - Detection rate (bare vs. sleeve)
    - Hash distance distributions (overall and by condition)
    - Confident vs. possible vs. miss rates
    - Per-hash-type breakdown (pHash, dHash, wHash)
    - Server vs. re-run comparison (consistency check)
    - OCR fallback estimate
    - Optional: CSV for spreadsheet analysis
    - Optional: saved intermediate crops for visual debugging
"""

import argparse
import csv
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Project imports (assumes running from repo root) ──
try:
    from image_match import (
        load_hash_database, match_card_image, crop_art_from_scan,
        compute_scan_hashes, DISTANCE_CONFIDENT, DISTANCE_POSSIBLE,
        HASH_WEIGHTS, _get_era, HASH_DB_FILE,
    )
    from card_detect import detect_and_crop_card
except ImportError as e:
    print(f"Import error: {e}")
    print("Run from the repo root: cd ~/card-scanner && python3 test_phone.py")
    sys.exit(1)

# ============================================
# Configuration
# ============================================
BASE_DIR = Path(__file__).parent
DEFAULT_TEST_DIR = BASE_DIR / "data" / "test_phone"
CROPS_DIR = BASE_DIR / "data" / "test_phone_crops"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_phone")


# ============================================
# Manifest Loading
# ============================================

def load_manifest(test_dir):
    """
    Load manifest.json from the test directory.

    Returns list of manifest entries, each with:
        seq, file, condition, timestamp, scan_result{...}
    """
    manifest_path = Path(test_dir) / "manifest.json"
    if not manifest_path.exists():
        logger.error("manifest.json not found in %s", test_dir)
        logger.info("Enable Test Mode in the mobile UI and scan some cards first.")
        return []

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("Failed to parse manifest.json: %s", e)
        return []

    if not isinstance(manifest, list):
        logger.error("manifest.json is not a list")
        return []

    # Validate that image files exist
    valid = []
    missing = 0
    for entry in manifest:
        img_path = Path(test_dir) / entry.get("file", "")
        if img_path.exists():
            entry["_path"] = img_path
            valid.append(entry)
        else:
            missing += 1

    if missing:
        logger.warning("%d manifest entries have missing image files", missing)

    return valid


# ============================================
# Single Image Test
# ============================================

def test_single_capture(entry, hash_db, save_crops=False):
    """
    Re-run the full detection + hash matching pipeline on one test capture.

    Compares fresh results against what the server originally recorded
    in the manifest to check consistency.

    Returns a result dict with all measurements.
    """
    path = entry["_path"]
    condition = entry.get("condition", "unknown")
    server_result = entry.get("scan_result", {})
    server_card_id = server_result.get("card_id")

    result = {
        "file": entry.get("file", path.name),
        "seq": entry.get("seq"),
        "condition": condition,
        "timestamp": entry.get("timestamp"),
        # Server's original identification
        "server_card_id": server_card_id,
        "server_name": server_result.get("name"),
        "server_id_method": server_result.get("id_method"),
        "server_hash_distance": server_result.get("hash_distance"),
        # Re-run detection stage
        "detect_success": False,
        "detect_w": None,
        "detect_h": None,
        "detect_aspect": None,
        "detect_time_ms": None,
        # Re-run hash matching stage
        "match_found": False,
        "match_id": None,
        "match_name": None,
        "match_distance": None,
        "match_confident": False,
        "match_time_ms": None,
        # Does re-run agree with server?
        "agrees_with_server": False,
        # Individual hash distances (for the best match)
        "phash_dist": None,
        "dhash_dist": None,
        "whash_dist": None,
        # Top-3 candidates
        "top3": [],
        # Error tracking
        "error": None,
    }

    # Load image
    img = cv2.imread(str(path))
    if img is None:
        result["error"] = "Failed to load image"
        return result

    # ── Stage 1: Card Detection ──
    t0 = time.perf_counter()
    detected, success = detect_and_crop_card(img)
    t1 = time.perf_counter()

    result["detect_success"] = success
    result["detect_time_ms"] = round((t1 - t0) * 1000, 1)

    if success:
        h, w = detected.shape[:2]
        result["detect_w"] = w
        result["detect_h"] = h
        result["detect_aspect"] = round(min(w, h) / max(w, h), 3)
    else:
        logger.debug("Detection failed for %s, using original image", path.name)

    # Save intermediate crop for debugging
    if save_crops:
        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        crop_name = f"{path.stem}_detected.jpg"
        cv2.imwrite(str(CROPS_DIR / crop_name), detected)

        # Also save the art crop (use default era since we don't know ground truth)
        art = crop_art_from_scan(detected, era="sm_swsh")
        art_bgr = cv2.cvtColor(np.array(art), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(CROPS_DIR / f"{path.stem}_art.jpg"), art_bgr)

    # ── Stage 2: Hash Matching ──
    t2 = time.perf_counter()
    match_result = match_card_image(detected, hash_db)
    t3 = time.perf_counter()

    result["match_time_ms"] = round((t3 - t2) * 1000, 1)
    result["match_found"] = match_result["match"]
    result["match_id"] = match_result["card_id"]
    result["match_name"] = match_result["name"]
    result["match_distance"] = match_result["distance"]
    result["match_confident"] = match_result["confident"]

    # Agreement with server
    result["agrees_with_server"] = (match_result["card_id"] == server_card_id)

    # Extract individual hash distances for top match
    candidates = match_result.get("candidates", [])
    if candidates:
        best = candidates[0]
        dists = best.get("distances", {})
        result["phash_dist"] = dists.get("phash")
        result["dhash_dist"] = dists.get("dhash")
        result["whash_dist"] = dists.get("whash")

    # Top-3 for analysis
    result["top3"] = [
        {
            "card_id": c["card_id"],
            "name": c["name"],
            "distance": c["distance"],
            "confident": c["confident"],
        }
        for c in candidates[:3]
    ]

    return result


# ============================================
# Report Generation
# ============================================

def print_report(results, hash_db):
    """Print a formatted analysis report."""
    total = len(results)
    if total == 0:
        print("\nNo test results to analyze.")
        return

    # Filter out errors
    valid = [r for r in results if not r["error"]]
    errors = [r for r in results if r["error"]]

    print(f"\n{'='*70}")
    print("PHASE 1 — PHONE HASH ROBUSTNESS REPORT")
    print(f"{'='*70}")
    print(f"  Total captures:   {total}")
    print(f"  Valid tests:      {len(valid)}")
    print(f"  Load errors:      {len(errors)}")
    print()

    if not valid:
        print("No valid test results to analyze.")
        return

    # ── Detection Stage ──
    detected = [r for r in valid if r["detect_success"]]
    detect_rate = len(detected) / len(valid) * 100
    detect_times = [r["detect_time_ms"] for r in valid if r["detect_time_ms"]]

    print(f"── CARD DETECTION ──")
    print(f"  Detection rate:   {len(detected)}/{len(valid)} ({detect_rate:.0f}%)")
    if detect_times:
        print(f"  Detect time:      median {statistics.median(detect_times):.0f}ms, "
              f"mean {statistics.mean(detect_times):.0f}ms")
    print()

    # ── Detection by condition ──
    conditions = sorted(set(r["condition"] for r in valid))
    if len(conditions) > 1:
        print(f"  Detection by condition:")
        for cond in conditions:
            cond_results = [r for r in valid if r["condition"] == cond]
            cond_detected = [r for r in cond_results if r["detect_success"]]
            pct = len(cond_detected) / len(cond_results) * 100 if cond_results else 0
            print(f"    {cond:12s}  {len(cond_detected)}/{len(cond_results)} ({pct:.0f}%)")
        print()

    # ── Hash Matching Stage ──
    matched = [r for r in valid if r["match_found"]]
    confident = [r for r in valid if r["match_confident"]]
    agrees = [r for r in valid if r["agrees_with_server"]]

    match_rate = len(matched) / len(valid) * 100
    confident_rate = len(confident) / len(valid) * 100
    agree_rate = len(agrees) / len(valid) * 100

    print(f"── HASH MATCHING ──")
    print(f"  Any match found:      {len(matched)}/{len(valid)} ({match_rate:.0f}%)")
    print(f"  Confident (d≤{DISTANCE_CONFIDENT}):    {len(confident)}/{len(valid)} ({confident_rate:.0f}%)")
    print(f"  Agrees with server:   {len(agrees)}/{len(valid)} ({agree_rate:.0f}%)")
    print()

    # ── Disagreements ──
    disagrees = [r for r in valid if not r["agrees_with_server"]]
    if disagrees:
        print(f"  ⚠ DISAGREEMENTS (re-run vs. server):")
        for r in disagrees[:10]:
            print(f"    {r['file']}: server={r['server_card_id']} "
                  f"rerun={r['match_id']} (d={r['match_distance']})")
        if len(disagrees) > 10:
            print(f"    ... and {len(disagrees) - 10} more")
        print()

    # ── Distance Distribution ──
    distances_all = [r["match_distance"] for r in valid if r["match_distance"] is not None]
    distances_confident = [r["match_distance"] for r in valid
                           if r["match_confident"] and r["match_distance"] is not None]

    print(f"── DISTANCE DISTRIBUTION (weighted) ──")
    if distances_all:
        print(f"  All matches:        n={len(distances_all)}")
        if len(distances_all) > 1:
            print(f"    min={min(distances_all):.1f}  "
                  f"median={statistics.median(distances_all):.1f}  "
                  f"mean={statistics.mean(distances_all):.1f}  "
                  f"max={max(distances_all):.1f}  "
                  f"stdev={statistics.stdev(distances_all):.1f}")
        else:
            print(f"    value={distances_all[0]:.1f}")

    if distances_confident:
        print(f"  Confident matches:  n={len(distances_confident)}")
        if len(distances_confident) > 1:
            print(f"    min={min(distances_confident):.1f}  "
                  f"median={statistics.median(distances_confident):.1f}  "
                  f"mean={statistics.mean(distances_confident):.1f}  "
                  f"max={max(distances_confident):.1f}")
    print()

    # ── Distance Buckets ──
    if distances_all:
        buckets = {"0-4": 0, "5-8": 0, "9-12": 0, "13-15": 0, ">15": 0}
        for d in distances_all:
            if d <= 4:
                buckets["0-4"] += 1
            elif d <= 8:
                buckets["5-8"] += 1
            elif d <= 12:
                buckets["9-12"] += 1
            elif d <= 15:
                buckets["13-15"] += 1
            else:
                buckets[">15"] += 1

        print(f"  Distance buckets (all matches):")
        for bucket, count in buckets.items():
            bar = "█" * count
            pct = count / len(distances_all) * 100
            zone = ""
            if bucket in ("0-4", "5-8"):
                zone = " ← CONFIDENT"
            elif bucket in ("9-12", "13-15"):
                zone = " ← falls to OCR"
            print(f"    {bucket:6s}  {count:3d} ({pct:5.1f}%) {bar}{zone}")
        print()

    # ── Per-Hash Type Breakdown ──
    phash_dists = [r["phash_dist"] for r in valid if r["phash_dist"] is not None]
    dhash_dists = [r["dhash_dist"] for r in valid if r["dhash_dist"] is not None]
    whash_dists = [r["whash_dist"] for r in valid if r["whash_dist"] is not None]

    if phash_dists:
        print(f"── PER-HASH DISTANCES (all matches) ──")
        for label, dists in [("pHash", phash_dists), ("dHash", dhash_dists),
                             ("wHash", whash_dists)]:
            if dists and len(dists) > 1:
                print(f"  {label}:  median={statistics.median(dists):.0f}  "
                      f"mean={statistics.mean(dists):.1f}  "
                      f"max={max(dists)}  "
                      f"stdev={statistics.stdev(dists):.1f}")
            elif dists:
                print(f"  {label}:  value={dists[0]}")
        print()

    # ── Accuracy by Condition ──
    if len(conditions) > 1:
        print(f"── RESULTS BY CONDITION ──")
        for cond in conditions:
            cond_valid = [r for r in valid if r["condition"] == cond]
            cond_detected = [r for r in cond_valid if r["detect_success"]]
            cond_matched = [r for r in cond_valid if r["match_found"]]
            cond_confident = [r for r in cond_valid if r["match_confident"]]
            cond_dists = [r["match_distance"] for r in cond_valid
                          if r["match_distance"] is not None]
            median_d = f"{statistics.median(cond_dists):.1f}" if cond_dists else "N/A"
            print(f"  {cond:12s}  detect={len(cond_detected)}/{len(cond_valid)}  "
                  f"match={len(cond_matched)}/{len(cond_valid)}  "
                  f"confident={len(cond_confident)}/{len(cond_valid)}  "
                  f"median_dist={median_d}")
        print()

    # ── OCR Fallback Estimate ──
    would_need_ocr = [r for r in valid if not r["match_confident"]]
    ocr_rate = len(would_need_ocr) / len(valid) * 100
    print(f"── OCR FALLBACK ESTIMATE ──")
    print(f"  Would fall through to OCR:  {len(would_need_ocr)}/{len(valid)} ({ocr_rate:.0f}%)")
    confident_count = len(valid) - len(would_need_ocr)
    print(f"  Would skip OCR (hash confident): "
          f"{confident_count}/{len(valid)} "
          f"({confident_count/len(valid)*100:.0f}%)")
    print()

    # ── Per-Image Detail ──
    print(f"── PER-IMAGE RESULTS ──")
    print(f"  {'File':<20s} {'Server ID':<14s} {'Re-run ID':<14s} {'Dist':>5s} "
          f"{'Conf':>4s} {'Agr':>3s} {'Det':>4s} {'Cond':<10s}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*5} {'-'*4} {'-'*3} {'-'*4} {'-'*10}")
    for r in results:
        if r["error"]:
            print(f"  {r['file']:<20s} {'ERROR':<14s} {'':>14s} "
                  f"{'':>5s} {'':>4s} {'':>3s} {'':>4s} {r['condition']:<10s}  {r['error']}")
            continue

        dist_str = f"{r['match_distance']:.1f}" if r["match_distance"] is not None else "—"
        conf_str = "YES" if r["match_confident"] else "no"
        agr_str = "✓" if r["agrees_with_server"] else "✗"
        det_str = "YES" if r["detect_success"] else "no"
        server_id = r["server_card_id"] or "none"
        match_id = r["match_id"] or "none"

        print(f"  {r['file']:<20s} {server_id:<14s} {match_id:<14s} "
              f"{dist_str:>5s} {conf_str:>4s} {agr_str:>3s} {det_str:>4s} {r['condition']:<10s}")

    # ── Key Takeaway ──
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    if distances_all:
        median_d = statistics.median(distances_all)
        if median_d <= DISTANCE_CONFIDENT:
            print(f"  ✓ Median distance ({median_d:.1f}) is within CONFIDENT threshold ({DISTANCE_CONFIDENT}).")
            print(f"    Hash-first identification is viable for phone photos.")
        elif median_d <= DISTANCE_POSSIBLE:
            print(f"  ⚠ Median distance ({median_d:.1f}) exceeds CONFIDENT ({DISTANCE_CONFIDENT})")
            print(f"    but is within POSSIBLE ({DISTANCE_POSSIBLE}).")
            print(f"    Options: lower confident threshold, add preprocessing, or")
            print(f"    use hash top-3 to narrow OCR search space.")
        else:
            print(f"  ✗ Median distance ({median_d:.1f}) exceeds POSSIBLE ({DISTANCE_POSSIBLE}).")
            print(f"    Perceptual hashing is not reliable for phone photos.")
            print(f"    Consider CLIP embeddings as replacement.")

    if ocr_rate > 60:
        print(f"  ⚠ {ocr_rate:.0f}% of scans would need OCR fallback.")
        print(f"    At 1-2s per OCR call, this significantly impacts scan speed.")

    # Condition comparison
    if len(conditions) > 1 and "bare" in conditions and "sleeve" in conditions:
        bare_dists = [r["match_distance"] for r in valid
                      if r["condition"] == "bare" and r["match_distance"] is not None]
        sleeve_dists = [r["match_distance"] for r in valid
                        if r["condition"] == "sleeve" and r["match_distance"] is not None]
        if bare_dists and sleeve_dists:
            bare_med = statistics.median(bare_dists)
            sleeve_med = statistics.median(sleeve_dists)
            delta = sleeve_med - bare_med
            print(f"\n  Bare median distance:   {bare_med:.1f}")
            print(f"  Sleeve median distance: {sleeve_med:.1f}")
            print(f"  Delta (sleeve - bare):  {delta:+.1f}")
            if delta > 3:
                print(f"  ⚠ Sleeves add significant hash distance. "
                      f"Card detection may be finding sleeve edge instead of card edge.")
            elif delta > 1:
                print(f"  ~ Sleeves add moderate hash distance. Acceptable if still within CONFIDENT.")
            else:
                print(f"  ✓ Sleeves have minimal impact on hash distance.")


def write_csv(results, output_path):
    """Write results to CSV for spreadsheet analysis."""
    if not results:
        return

    fieldnames = [
        "file", "seq", "condition", "timestamp",
        "server_card_id", "server_name", "server_id_method", "server_hash_distance",
        "detect_success", "detect_w", "detect_h", "detect_aspect", "detect_time_ms",
        "match_found", "match_id", "match_name", "match_distance",
        "match_confident", "agrees_with_server", "match_time_ms",
        "phash_dist", "dhash_dist", "whash_dist",
        "error",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    logger.info("CSV saved: %s", output_path)


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Measure hash robustness on phone camera photos"
    )
    parser.add_argument(
        "--dir", type=str, default=str(DEFAULT_TEST_DIR),
        help=f"Directory containing manifest.json and captures (default: {DEFAULT_TEST_DIR})"
    )
    parser.add_argument(
        "--save-crops", action="store_true",
        help="Save detected card and art crop images for visual inspection"
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Output results as CSV alongside the printed report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-image progress during testing"
    )
    parser.add_argument(
        "--list-ids", type=str, default=None, nargs="?", const="",
        help="List card IDs in hash DB. Optional prefix filter (e.g. --list-ids sv)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 1 — Phone Hash Robustness Test")
    print("=" * 70)
    print()

    # ── List IDs mode ──
    if args.list_ids is not None:
        print("Loading hash database...")
        hash_db_raw = None
        if HASH_DB_FILE.exists():
            with open(HASH_DB_FILE, "r") as f:
                hash_db_raw = json.load(f)
        if not hash_db_raw or "hashes" not in hash_db_raw:
            print("ERROR: Hash database not found or empty.")
            sys.exit(1)

        hashes = hash_db_raw["hashes"]
        prefix = args.list_ids.strip().lower()
        ids = sorted(hashes.keys())
        if prefix:
            ids = [i for i in ids if i.lower().startswith(prefix)]

        print(f"  {len(hashes)} total cards in DB, showing {len(ids)} "
              f"{'matching \"' + prefix + '\"' if prefix else '(all)'}\n")

        # Group by set prefix
        by_set = {}
        for cid in ids:
            set_id = hashes[cid].get("set_id", cid.rsplit("-", 1)[0])
            by_set.setdefault(set_id, []).append(cid)

        for set_id in sorted(by_set.keys()):
            cards = by_set[set_id]
            print(f"  {set_id} ({len(cards)} cards):")
            for cid in cards[:10]:
                print(f"    {cid:<20s} {hashes[cid]['name']}")
            if len(cards) > 10:
                print(f"    ... and {len(cards) - 10} more")
            print()

        print("Scan cards in Test Mode and they'll be logged to manifest.json.")
        sys.exit(0)

    # ── Load hash database ──
    print("Loading hash database...")
    hash_db = load_hash_database()
    if hash_db is None:
        print("ERROR: Hash database not found or empty.")
        print("Run build_hash_db.py first.")
        sys.exit(1)
    print(f"  {len(hash_db)} cards in hash database")
    print()

    # ── Load manifest ──
    print(f"Loading manifest from: {args.dir}")
    manifest = load_manifest(args.dir)
    if not manifest:
        print(f"\nNo test captures found in {args.dir}")
        print(f"\nTo get started:")
        print(f"  1. Open the mobile scanner UI: http://<jetson-ip>:8080/mobile")
        print(f"  2. Tap 'Test' to enable Test Mode")
        print(f"  3. Set condition (Bare/Sleeve) and scan cards")
        print(f"  4. Re-run this script")
        sys.exit(0)

    conditions = set(e.get("condition", "unknown") for e in manifest)
    identified = sum(1 for e in manifest if e.get("scan_result", {}).get("card_id"))
    print(f"  Found {len(manifest)} captures")
    print(f"  Conditions: {', '.join(sorted(conditions))}")
    print(f"  Server identified: {identified}/{len(manifest)}")
    print()

    # ── Run tests ──
    print("Re-running detection + hashing on all captures...")
    results = []
    t_start = time.time()

    for i, entry in enumerate(manifest):
        if args.verbose:
            print(f"  [{i+1}/{len(manifest)}] {entry.get('file', '?')} ", end="", flush=True)

        result = test_single_capture(entry, hash_db, save_crops=args.save_crops)
        results.append(result)

        if args.verbose:
            agr = "✓" if result["agrees_with_server"] else "✗"
            dist = f"d={result['match_distance']:.1f}" if result["match_distance"] else "no match"
            print(f"→ {agr} {dist}")

    elapsed = time.time() - t_start
    if elapsed > 0:
        print(f"  Completed {len(results)} tests in {elapsed:.1f}s "
              f"({len(results)/elapsed:.1f} tests/sec)")

    # ── Output ──
    print_report(results, hash_db)

    if args.csv:
        csv_path = Path(args.dir) / "results.csv"
        write_csv(results, csv_path)

    if args.save_crops:
        print(f"\nIntermediate crops saved to: {CROPS_DIR}")
        print("  *_detected.jpg  — perspective-corrected card (from card_detect.py)")
        print("  *_art.jpg       — cropped art region (input to hash computation)")


if __name__ == "__main__":
    main()

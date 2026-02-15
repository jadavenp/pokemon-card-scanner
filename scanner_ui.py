"""
scanner_ui.py
Visual scanner UI — generates compact HTML results and opens in browser.

Can be called standalone or imported by test_random.py.

Usage:
    python3 scanner_ui.py                  # scan data/ images + open browser
    python3 scanner_ui.py --verbose        # extra debug in terminal
    python3 scanner_ui.py --no-hash        # hash disabled
    python3 scanner_ui.py --no-open        # don't auto-open browser
"""

import argparse
import base64
import os
import sys
import time
import webbrowser
from pathlib import Path

from scanner import process_image, find_card_images
from config import IMAGE_DIR, SET_NAMES, DATA_DIR, VARIANT_DISPLAY
from database import load_database
from stamp import load_stamp_template

try:
    from image_match import load_hash_database
    IMAGE_MATCH_AVAILABLE = True
except ImportError:
    IMAGE_MATCH_AVAILABLE = False


def image_to_base64(img_path):
    """Convert an image file to base64 data URI for embedding in HTML."""
    ext = Path(img_path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "gif": "image/gif", "webp": "image/webp"}.get(ext.lstrip("."), "image/png")
    try:
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{data}"
    except Exception:
        return ""


def generate_html(results, total_time):
    """Generate compact results HTML page."""

    # Aggregate stats
    identified = sum(1 for r in results if not r.get("error"))
    failed = sum(1 for r in results if r.get("error"))
    first_eds = sum(1 for r in results if r.get("stamp_1st"))
    total_value = 0.0
    hash_ids = ocr_ids = hash_ocr_ids = 0
    for r in results:
        if r.get("price", "").startswith("$"):
            try:
                total_value += float(r["price"][1:])
            except ValueError:
                pass
        m = r.get("id_method", "")
        if m == "hash":
            hash_ids += 1
        elif m == "ocr+hash":
            hash_ocr_ids += 1
        elif m in ("ocr", "ocr (hash disagree)"):
            ocr_ids += 1

    # Build card rows
    card_rows = ""
    for r in results:
        img_path = Path(IMAGE_DIR) / r["file"]
        # Also check with _test_ prefix for test_random symlinks
        if not img_path.exists():
            img_path = Path(IMAGE_DIR) / f"_test_{r['file']}"
        # Also check Ref Images directly by card_id
        if not img_path.exists() and r.get("card_id"):
            img_path = Path(DATA_DIR) / "Ref Images" / f"{r['card_id']}.png"
        img_data = image_to_base64(img_path)

        if r.get("error"):
            card_rows += f'''
            <tr class="error-row">
                <td class="img-cell"><div class="img-wrap"><img src="{img_data}" alt=""></div></td>
                <td colspan="8" class="error-text">ERROR: {r["error"]}</td>
                <td class="time-cell">{r.get("time", 0):.1f}s</td>
            </tr>'''
            continue

        name = r.get("name") or "?"
        set_name = r.get("set_name") or "?"
        number = r.get("number") or "?"
        total = r.get("total") or "?"
        edition = "1st" if r.get("stamp_1st") else "Unl"
        ed_class = "ed-1st" if r.get("stamp_1st") else "ed-unl"
        price = r.get("price", "N/A")
        variant = VARIANT_DISPLAY.get(r.get("price_variant", ""), r.get("price_variant", ""))
        rarity = r.get("rarity", "?")
        method = r.get("id_method", "?")
        elapsed = r.get("time", 0)
        hash_dist = r.get("hash_distance")
        name_conf = r.get("name_conf", 0)
        num_conf = r.get("num_conf", 0)
        stamp_conf = r.get("stamp_conf", 0)

        # Method badge color
        method_colors = {
            "hash": "#22c55e",
            "ocr+hash": "#3b82f6",
            "ocr": "#f97316",
            "ocr (hash disagree)": "#ef4444",
        }
        method_labels = {
            "hash": "HASH",
            "ocr+hash": "OCR+H",
            "ocr": "OCR",
            "ocr (hash disagree)": "OCR!H",
        }
        m_color = method_colors.get(method, "#666")
        m_label = method_labels.get(method, method or "?")

        hash_str = f"{hash_dist}" if hash_dist is not None else "—"

        card_rows += f'''
            <tr>
                <td class="img-cell"><div class="img-wrap"><img src="{img_data}" alt="{name}"></div></td>
                <td class="name-cell">
                    <span class="card-name">{name}</span>
                    <span class="card-set">{set_name}</span>
                </td>
                <td class="num-cell">{number}/{total}</td>
                <td class="rarity-cell">{rarity}</td>
                <td class="{ed_class}">{edition}</td>
                <td class="price-cell">{price}<span class="variant">{variant}</span></td>
                <td class="method-cell"><span class="badge" style="background:{m_color}">{m_label}</span><span class="hash-dist">{hash_str}</span></td>
                <td class="conf-cell">
                    <div class="conf-row"><span class="cl">N</span><div class="bar"><div class="fill" style="width:{name_conf}%"></div></div><span class="cp">{name_conf:.0f}</span></div>
                    <div class="conf-row"><span class="cl">#</span><div class="bar"><div class="fill" style="width:{num_conf}%"></div></div><span class="cp">{num_conf:.0f}</span></div>
                    <div class="conf-row"><span class="cl">S</span><div class="bar"><div class="fill" style="width:{min(stamp_conf, 100)}%"></div></div><span class="cp">{stamp_conf:.0f}</span></div>
                </td>
                <td class="time-cell">{elapsed:.1f}s</td>
            </tr>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pokemon Card Scanner</title>
<style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: -apple-system, 'Segoe UI', sans-serif; background:#0a0a0a; color:#ccc; font-size:13px; }}

    .header {{
        display:flex; justify-content:space-between; align-items:center;
        padding:12px 20px; background:#111; border-bottom:2px solid #e2b714;
    }}
    .header h1 {{ font-size:15px; color:#e2b714; letter-spacing:1px; text-transform:uppercase; }}
    .header .sub {{ font-size:11px; color:#555; font-family:monospace; }}

    .stats {{
        display:flex; gap:24px; padding:8px 20px; background:#0f0f0f;
        border-bottom:1px solid #1a1a1a; flex-wrap:wrap;
    }}
    .stat {{ display:flex; gap:6px; align-items:baseline; }}
    .stat .label {{ font-size:10px; text-transform:uppercase; letter-spacing:.5px; color:#555; }}
    .stat .val {{ font-family:monospace; font-size:14px; font-weight:700; color:#e2b714; }}
    .stat .val.g {{ color:#22c55e; }}
    .stat .val.r {{ color:#ef4444; }}

    table {{ width:100%; border-collapse:collapse; }}
    thead th {{
        position:sticky; top:0; background:#111; padding:6px 10px;
        font-size:10px; text-transform:uppercase; letter-spacing:.5px;
        color:#555; text-align:left; border-bottom:1px solid #222; z-index:10;
    }}
    tbody tr {{ border-bottom:1px solid #1a1a1a; }}
    tbody tr:hover {{ background:#151515; }}
    td {{ padding:6px 10px; vertical-align:middle; }}

    .img-cell {{ width:60px; padding:4px 8px; }}
    .img-wrap {{ width:48px; }}
    .img-wrap img {{ width:100%; border-radius:3px; display:block; }}

    .name-cell {{ min-width:140px; }}
    .card-name {{ display:block; font-weight:600; color:#fff; font-size:13px; line-height:1.2; }}
    .card-set {{ display:block; font-size:10px; color:#666; }}

    .num-cell {{ font-family:monospace; font-size:12px; color:#999; white-space:nowrap; }}
    .rarity-cell {{ font-size:11px; color:#888; }}

    .ed-1st {{ font-weight:700; color:#e2b714; font-size:12px; }}
    .ed-unl {{ color:#555; font-size:12px; }}

    .price-cell {{ font-family:monospace; font-weight:700; color:#22c55e; font-size:13px; white-space:nowrap; }}
    .price-cell .variant {{ font-size:9px; color:#555; font-weight:400; margin-left:4px; }}

    .method-cell {{ white-space:nowrap; }}
    .badge {{
        display:inline-block; font-family:monospace; font-size:10px; font-weight:700;
        padding:2px 5px; border-radius:3px; color:#000; letter-spacing:.3px;
    }}
    .hash-dist {{ font-family:monospace; font-size:10px; color:#555; margin-left:4px; }}

    .conf-cell {{ width:120px; }}
    .conf-row {{ display:flex; align-items:center; gap:3px; height:12px; }}
    .cl {{ font-size:9px; color:#444; width:10px; text-align:right; }}
    .bar {{ flex:1; height:3px; background:#222; border-radius:1px; overflow:hidden; }}
    .fill {{ height:100%; background:#e2b714; border-radius:1px; }}
    .cp {{ font-family:monospace; font-size:9px; color:#444; width:22px; text-align:right; }}

    .time-cell {{ font-family:monospace; font-size:11px; color:#444; text-align:right; white-space:nowrap; }}

    .error-row td {{ color:#ef4444; }}
    .error-text {{ font-family:monospace; font-size:12px; }}

    .footer {{
        padding:10px 20px; border-top:1px solid #1a1a1a;
        font-size:10px; color:#333; text-align:center; font-family:monospace;
    }}

    /* Lightbox overlay */
    .lightbox {{
        display:none; position:fixed; top:0; left:0; width:100%; height:100%;
        background:rgba(0,0,0,0.85); z-index:100; cursor:pointer;
        justify-content:center; align-items:center;
    }}
    .lightbox.active {{ display:flex; }}
    .lightbox img {{
        max-height:90vh; max-width:90vw; border-radius:8px;
        box-shadow:0 0 40px rgba(0,0,0,0.8);
    }}
    .img-wrap {{ cursor:pointer; }}
    .img-wrap:hover {{ opacity:0.8; }}
</style>
</head>
<body>
    <div class="header">
        <h1>Pokemon Card Scanner</h1>
        <span class="sub">Hash → OCR → Stamp → Price</span>
    </div>

    <div class="stats">
        <div class="stat"><span class="label">Scanned</span><span class="val">{len(results)}</span></div>
        <div class="stat"><span class="label">ID'd</span><span class="val g">{identified}</span></div>
        <div class="stat"><span class="label">Failed</span><span class="val{' r' if failed else ''}">{failed}</span></div>
        <div class="stat"><span class="label">1st Ed</span><span class="val">{first_eds}</span></div>
        <div class="stat"><span class="label">Value</span><span class="val g">${total_value:.2f}</span></div>
        <div class="stat"><span class="label">Hash</span><span class="val">{hash_ids}</span></div>
        <div class="stat"><span class="label">OCR+H</span><span class="val">{hash_ocr_ids}</span></div>
        <div class="stat"><span class="label">OCR</span><span class="val">{ocr_ids}</span></div>
        <div class="stat"><span class="label">Time</span><span class="val">{total_time:.1f}s</span></div>
    </div>

    <table>
        <thead>
            <tr>
                <th></th>
                <th>Card</th>
                <th>Num</th>
                <th>Rarity</th>
                <th>Ed</th>
                <th>Price</th>
                <th>Method</th>
                <th>Confidence</th>
                <th style="text-align:right">Time</th>
            </tr>
        </thead>
        <tbody>
            {card_rows}
        </tbody>
    </table>

    <div class="footer">
        Generated {time.strftime("%Y-%m-%d %H:%M:%S")} · {len(results)} cards · {total_time:.1f}s
    </div>

    <div class="lightbox" id="lightbox" onclick="this.classList.remove('active')">
        <img id="lightbox-img" src="" alt="">
    </div>

    <script>
        document.querySelectorAll('.img-wrap img').forEach(img => {{
            img.addEventListener('click', () => {{
                document.getElementById('lightbox-img').src = img.src;
                document.getElementById('lightbox').classList.add('active');
            }});
        }});
        document.addEventListener('keydown', e => {{
            if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('active');
        }});
    </script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description="Pokemon Card Scanner — Visual UI")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-hash", action="store_true")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open browser")
    args = parser.parse_args()

    overall_start = time.time()

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

    # Find and process images
    image_files = find_card_images()
    if not image_files:
        print(f"\nNo images found in {IMAGE_DIR}")
        sys.exit(1)

    print(f"\nScanning {len(image_files)} card(s)...")

    results = []
    for i, img_path in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] {img_path.name}...", end=" ", flush=True)
        t0 = time.time()
        r = process_image(img_path, reader, index, stamp_template,
                          hash_db=hash_db, verbose=args.verbose)
        elapsed = time.time() - t0
        r["time"] = elapsed

        name = r.get("name", "?")
        method = r.get("id_method", "?")
        print(f"{name} [{method}] ({elapsed:.1f}s)")

        results.append(r)

    total_time = time.time() - overall_start

    # Generate and save HTML
    html = generate_html(results, total_time)
    output_path = Path(DATA_DIR) / "scan_results.html"
    with open(output_path, "w") as f:
        f.write(html)

    print(f"\nResults saved to: {output_path}")

    if not args.no_open:
        webbrowser.open(f"file://{output_path.resolve()}")
        print("Opened in browser.")

    identified = sum(1 for r in results if not r.get("error"))
    total_value = sum(
        float(r["price"][1:]) for r in results
        if r.get("price", "").startswith("$")
    )
    print(f"\n  {identified}/{len(results)} identified | ${total_value:.2f} total | {total_time:.1f}s")


if __name__ == "__main__":
    main()

# Phase 1 Test — Workflow Checklist

## Setup (one-time)

1. Drop the updated files into `~/card-scanner/`:
   - `server.py` (replaces existing)
   - `scanner_mobile.html` (replaces existing)
   - `test_phone.py` (replaces existing)
   - `start_phase1_test.sh` (new)

2. Make the startup script executable:
   ```
   cd ~/card-scanner
   chmod +x start_phase1_test.sh
   ```

## Run the Server

```bash
# On the Jetson (LAN-accessible, CSI camera):
cd ~/card-scanner
./start_phase1_test.sh

# On Mac (dev/mock mode):
cd ~/card-scanner
./start_phase1_test.sh --mock
```

The script activates the venv, checks dependencies, and starts the server.
It prints the URL to open on your phone.

## Collect Test Data

1. Open `http://<jetson-ip>:8080/mobile` on your phone
2. Tap **Test** (right of the SCAN button) — it lights up gold
3. A condition bar appears: **Bare** / **Sleeve** / **Toploader**
4. Set condition, point at a card, tap **SCAN**
5. Toast confirms: `"Test saved: capture_001.jpg"`
6. Repeat — flip between Bare and Sleeve as you go
7. When Test is OFF, scans work normally (no data saved)

**Target:** 20-40 cards, roughly half bare / half sleeved. Modern cards
are fine. Use real shop lighting — fluorescent overhead, counter surface.

### Check progress (optional)

From any browser:
```
http://<jetson-ip>:8080/test/captures
```
Returns JSON with capture count and condition breakdown.

## Analyze Results

```bash
cd ~/card-scanner
source venv/bin/activate

# Standard report:
python3 test_phone.py

# With debug crops saved:
python3 test_phone.py --save-crops

# Export CSV for spreadsheet:
python3 test_phone.py --csv

# Verbose (per-image progress):
python3 test_phone.py --verbose

# Browse hash DB (what card IDs are available):
python3 test_phone.py --list-ids
python3 test_phone.py --list-ids base1
```

## What the Report Tells You

| Section | Question it answers |
|---------|-------------------|
| CARD DETECTION | Does card_detect.py find the card? Bare vs. sleeve? |
| HASH MATCHING | What % get a confident hash match (d ≤ 8)? |
| DISTANCE DISTRIBUTION | Where do phone photos land on the distance scale? |
| DISTANCE BUCKETS | Visual breakdown: how many in 0-4, 5-8, 9-12, etc. |
| PER-HASH DISTANCES | Which hash type degrades most on phone photos? |
| RESULTS BY CONDITION | Side-by-side bare vs. sleeve performance |
| OCR FALLBACK ESTIMATE | What % would need OCR (slow path)? |
| KEY FINDINGS | Go/no-go verdict on hashing viability |

## Decision Points Based on Results

- **Median distance ≤ 8, sleeve delta < 2** → Hashing works. Proceed with current pipeline.
- **Median distance 9-12** → Hashing is marginal. Consider preprocessing improvements or CLIP.
- **Median distance > 15** → Hashing doesn't transfer to phone photos. Move to CLIP embeddings.
- **Detection rate drops for sleeves** → card_detect.py is finding the sleeve edge. Prioritize YOLO.
- **OCR fallback > 60%** → Too many scans need the slow path. Hash pipeline needs improvement first.

## Data Location

```
data/test_phone/
├── manifest.json          # All scan results + metadata
├── capture_001.jpg        # Raw phone uploads
├── capture_002.jpg
└── ...

data/test_phone_crops/     # Only if --save-crops used
├── capture_001_detected.jpg
├── capture_001_art.jpg
└── ...
```

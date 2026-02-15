# Fix Reference — Before/After for Each Change

## Fix #1: scanner.py line 13 — Pricing Import

```diff
- from pricing import fetch_live_pricing
+ from pricing_justtcg import fetch_live_pricing
```

**Why:** `pricing.py` is the legacy TCGGO/RapidAPI module. JustTCG is the sole runtime pricing backend per project rules. `pricing_justtcg.py` exports the same `fetch_live_pricing(card_id, is_1st_edition, rarity)` signature, so this is a drop-in swap.

---

## Fix #2: scanner.py — Path conversion in process_image()

```diff
  def process_image(img_path, reader, index, stamp_template, hash_db=None, verbose=False):
+     img_path = Path(img_path)
      img = cv2.imread(str(img_path))
      if img is None:
          return {"file": img_path.name, "error": "Could not load image"}
```

**Why:** `server.py` passes `img_path` as a plain string from `cam.current_mock_path`. The line `img_path.name` is a `pathlib.Path` attribute. `Path()` accepts both strings and Path objects, so this is backward-compatible with batch mode.

---

## Fix #3: scanner.py line ~233 — Batch summary label

```diff
- print(f"  API calls:        TCGGO: {api_calls} / 100 daily")
+ print(f"  API calls:        JustTCG: {api_calls} / 100 daily")
```

**Why:** Cosmetic. Pricing comes from JustTCG now, not TCGGO.

---

## Fix #4: server.py — load_database() tuple unpacking

```diff
- scanner_resources["index"] = load_database()
+ scanner_resources["index"], _ = load_database()
```

**Why:** `load_database()` returns `(index_dict, card_count)`. Without unpacking, the entire tuple gets passed to `process_image()` → `lookup_card()`, which expects a dict.

---

## Fix #5: config.py — Database filename (conditional)

```diff
- DATABASE_FILE = DATA_DIR / "pokemon_tcg_database.json"
+ DATABASE_FILE = DATA_DIR / "card_index.json"
```

**Why:** Only applied if `data/card_index.json` exists and `data/pokemon_tcg_database.json` does not. The session doc references `card_index.json` as the 73MB, 20,078-card database file.

**Action:** The script checks which file exists and only patches if needed.

---

## Fix #6: scanner_dashboard.html — Error handling + null-safe time

**6a:** Error card guard added at the top of `buildCardRow()`:
```javascript
function buildCardRow(card, isNew) {
    // ── Error card guard ──
    if (card.error && !card.name) {
      return `
        <div class="scan-card ${isNew ? 'new-entry' : ''}">
          <div class="scan-thumb"><span class="scan-thumb-placeholder">⚠️</span></div>
          <div class="scan-info">
            <div class="scan-name" style="color:var(--accent-red)">Scan Error</div>
            <div class="scan-meta">${card.error}</div>
            <div class="scan-meta-row"><span style="font-size:10px;color:var(--text-muted)">${(card.time || 0).toFixed(1)}s</span></div>
          </div>
          <div class="scan-price"><div class="scan-price-value">—</div></div>
        </div>`;
    }
    // ... rest of existing function unchanged ...
```

**6b:** Null-safe time in normal card rows:
```diff
- ${card.time.toFixed(1)}s
+ ${(card.time || 0).toFixed(1)}s
```

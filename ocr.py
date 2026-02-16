"""
ocr.py — Card text extraction via EasyOCR.

Refactored (Session 4) to use targeted ROI crops instead of full-image scoring:
  - extract_name_candidates(): NEW — crops top 8% of card for name OCR
  - extract_hp(): NEW — extracts HP from top-right of card
  - extract_name(): PRESERVED — legacy full-image scoring (kept as fallback)
  - extract_number(): PRESERVED — consensus scoring from bottom strip
  - detect_card_type(): PRESERVED — banner/HP classification

Key changes:
  - Name extraction uses a dedicated ROI (y=0–0.08, x=0.05–0.75) instead of
    scoring all 28+ OCR hits from the full card image. This eliminates body text
    (attack names, abilities, flavor text) from competing with the actual name.
  - HP extraction pulls the numeric value from the top-right corner, providing
    a strong cross-referencing signal for database matching.
  - Both functions return multiple candidates for downstream multi-signal matching.
"""

import re
import cv2
import numpy as np

from pathlib import Path
import logging
import time

logger = logging.getLogger("ocr")

from config import (
    NAME_SKIP_WORDS, NAME_POSITION_WEIGHT, NAME_CONFIDENCE_WEIGHT,
    NAME_SIZE_WEIGHT, NAME_MIN_CONFIDENCE, NAME_MAX_Y_RATIO,
    NUMBER_CROP_Y, NUMBER_CROP_X, NUMBER_UPSCALE_LARGE,
    NUMBER_UPSCALE_SMALL, NUMBER_EARLY_EXIT_CONF,

    NUMBER_REGIONS, NUMBER_CLAHE_CLIP, NUMBER_CLAHE_GRID,
    NUMBER_HIGHLIGHT_CLAMP, NUMBER_UNSHARP_SIGMA, NUMBER_UNSHARP_WEIGHT,
    NUMBER_UNSHARP_BLUR_WEIGHT, NUMBER_BILATERAL_D, NUMBER_BILATERAL_SIGMA,
    NUMBER_ALLOWLIST, NUMBER_CONTRAST_THS, NUMBER_ADJUST_CONTRAST,
    NUMBER_MIN_SIZE, NUMBER_UPSCALE_V2, OCR_DEBUG_DIR,
    KNOWN_SET_TOTALS,
)

# ─────────────────────────────────────────────────────────────
# ROI REGIONS — card anatomy coordinates (fraction of card dims)
# ─────────────────────────────────────────────────────────────

# Name sits at the very top of the card, but position varies by card type:
# - Pokemon: name is on the first line (y=0–8%), alongside HP
# - Trainer: "Supporter/TRAINER" banner is line 1, actual name is line 2 (y=0–15%)
# - Energy: similar to Trainer, name below the type banner
NAME_ROIS = {
    "pokemon": {"y_start": 0.00, "y_end": 0.09, "x_start": 0.05, "x_end": 0.78},
    "trainer": {"y_start": 0.00, "y_end": 0.16, "x_start": 0.02, "x_end": 0.85},
    "energy":  {"y_start": 0.00, "y_end": 0.16, "x_start": 0.02, "x_end": 0.85},
}
NAME_ROI_DEFAULT = {"y_start": 0.00, "y_end": 0.12, "x_start": 0.02, "x_end": 0.85}

# HP is top-right corner, same vertical band as the name.
# Format on cards: "150 HP" or "HP 150" — always large, high-confidence text.
HP_ROI = {"y_start": 0.00, "y_end": 0.09, "x_start": 0.55, "x_end": 0.98}

# Minimum upscale for name/HP ROI before OCR (ensures text is large enough)
NAME_ROI_MIN_WIDTH = 600


# ─────────────────────────────────────────────────────────────
# CARD TYPE DETECTION (unchanged)
# ─────────────────────────────────────────────────────────────

def detect_card_type(ocr_results):
    """
    Determine if card is Pokemon, Trainer, or Energy.
    Uses priority logic to avoid misclassification:
      - "trainer" in large top text = Trainer
      - HP value or stage indicator present = Pokemon (even if "energy" in body)
      - "energy" in large top text with no HP/stage = Energy
    """
    all_text = " ".join([text.lower() for (bbox, text, conf) in ocr_results])

    has_hp = bool(re.search(r'\d+\s*hp', all_text))
    has_stage = any(s in all_text for s in [
        'stage 1', 'stage 2', 'basic pokémon', 'basic pokemon', 'evolves from'
    ])
    has_trainer_banner = False
    has_energy_banner = False

    for (bbox, text, conf) in ocr_results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        img_height = max(b[2][1] for b in [r[0] for r in ocr_results]) if ocr_results else 1
        if img_height > 0 and (y_center / img_height) < 0.25:
            if "trainer" in text.lower():
                has_trainer_banner = True
            if "energy" in text.lower():
                has_energy_banner = True

    if has_trainer_banner:
        return "trainer"
    if has_hp or has_stage:
        return "pokemon"
    if has_energy_banner or "energy" in all_text:
        return "energy"
    return "pokemon"


# ─────────────────────────────────────────────────────────────
# NAME EXTRACTION — DUAL-SOURCE (full-image + ROI boost)
# ─────────────────────────────────────────────────────────────

# Words that appear on cards but are never the card name.
# Includes common OCR misreads of banner text.
_NAME_REJECT_EXACT = set(NAME_SKIP_WORDS) | {
    'supporter', 'trainer', 'tra', 'traingr', 'item', 'stadium', 'tool',
    'energy', 'basic', 'stage', 'pokemon', 'pokémon',
    'rule', 'rules', 'v', 'vmax', 'vstar', 'gx', 'ex',
    'lv.x', 'break', 'radiant', 'ancient', 'future',
}

# Prefixes of common banner words — catches OCR garbled variants like
# "Supported", "Suppceter", "TRAINEF", "TPAINER", "TpA", etc.
# If a short candidate (<=10 chars) starts with one of these, reject it.
_BANNER_PREFIXES = (
    'support', 'supp', 'train', 'trai',
    'stadi', 'energi', 'energ',
)

# Known OCR garbled readings of banner words that consistently appear
_NAME_REJECT_GARBLED = {
    'supported', 'suppceter', 'suppcrter', 'supparter', 'supponter',
    'trainef', 'tpainer', 'traingr', 'traner', 'trainef',
    'tpa', 'tl', 'tra', 'tra1ner',
    'dasic', 'basig', 'basi', 'stagei', 'stage1',
}

_NAME_REJECT = _NAME_REJECT_EXACT | _NAME_REJECT_GARBLED


def extract_name_candidates(img, reader, card_type="pokemon",
                            ocr_results=None, max_candidates=5):
    """
    Extract card name candidates from two sources and merge the best:

    Source 1 — Full-image OCR (already computed for card type detection):
      Scores candidates from the top 20% of the card. Good at catching
      names the ROI crop misses (e.g. Trainer names at y=11%).

    Source 2 — ROI crop OCR (targeted strip at top of card):
      Card-type-specific crop. Higher confidence due to focused input.
      Pokemon: y=0–9%, Trainer/Energy: y=0–16%.

    Both sources' candidates are merged and deduplicated. If the same
    text appears in both, the higher score wins.

    Args:
        img: BGR card image (perspective-corrected)
        reader: EasyOCR reader instance
        card_type: "pokemon", "trainer", or "energy"
        ocr_results: pre-existing full-image OCR results (avoids redundant call)
        max_candidates: max candidates to return

    Returns:
        list of dicts sorted by score (best first):
        [{"name": str, "confidence": float, "score": float, "source": str}, ...]
    """
    h, w = img.shape[:2]
    candidates = {}  # name_lower -> best candidate dict

    # ── Source 1: Full-image OCR candidates (top 20% only) ──
    if ocr_results:
        max_y = max(max(pt[1] for pt in bbox) for (bbox, text, conf) in ocr_results)
        if max_y == 0:
            max_y = 1

        for (bbox, text, conf) in ocr_results:
            text_clean = text.strip()
            text_lower = text_clean.lower()

            # Position filter: name must be in top 20% of card
            ys = [pt[1] for pt in bbox]
            y_center = (min(ys) + max(ys)) / 2
            y_ratio = y_center / max_y
            if y_ratio > 0.20:
                continue

            if not _passes_name_filters(text_clean, text_lower, conf, min_conf=0.25):
                continue

            # Score: confidence + position (top = better) + bbox width
            xs = [pt[0] for pt in bbox]
            bbox_width = max(xs) - min(xs)
            width_ratio = bbox_width / w
            position_score = max(0, 1.0 - (y_ratio / 0.20))

            score = (conf * 0.50) + (position_score * 0.20) + (width_ratio * 0.30)

            _update_candidate(candidates, text_clean, conf, score, "full_ocr")

    # ── Source 2: ROI crop OCR candidates ──
    name_roi = NAME_ROIS.get(card_type, NAME_ROI_DEFAULT)
    y0 = int(h * name_roi["y_start"])
    y1 = int(h * name_roi["y_end"])
    x0 = int(w * name_roi["x_start"])
    x1 = int(w * name_roi["x_end"])

    roi = img[y0:y1, x0:x1]
    if roi.size > 0:
        roi_w = roi.shape[1]
        if roi_w < NAME_ROI_MIN_WIDTH:
            scale = NAME_ROI_MIN_WIDTH / roi_w
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        _save_debug_image(roi, "name_roi", card_type)

        try:
            roi_results = reader.readtext(roi)
            roi_h, roi_w = roi.shape[:2]

            logger.info("Name ROI OCR (%d items from %dx%d crop, type=%s):",
                        len(roi_results), roi_w, roi_h, card_type)

            for (bbox, text, conf) in roi_results:
                text_clean = text.strip()
                text_lower = text_clean.lower()

                logger.info("  roi_text='%s' conf=%.2f", text_clean, conf)

                if not _passes_name_filters(text_clean, text_lower, conf, min_conf=0.15):
                    continue

                xs = [pt[0] for pt in bbox]
                bbox_width = max(xs) - min(xs)
                width_ratio = bbox_width / roi_w

                # ROI gets a small boost — focused crop means less noise
                score = (conf * 1.05 * 0.65) + (width_ratio * 0.35)

                _update_candidate(candidates, text_clean, conf, score, "roi")

        except Exception as e:
            logger.warning("EasyOCR failed on name ROI: %s", e)

    # ── Merge, sort, return ──
    result = sorted(candidates.values(), key=lambda x: x["score"], reverse=True)
    result = result[:max_candidates]

    for i, c in enumerate(result):
        logger.info("  Name candidate #%d: '%s' conf=%.1f%% score=%.3f source=%s",
                     i + 1, c["name"], c["confidence"], c["score"], c["source"])

    if not result:
        logger.info("  No name candidates found from either source")

    return result


def _passes_name_filters(text_clean, text_lower, conf, min_conf=0.25):
    """Shared filter logic for name candidates from any source."""
    if len(text_clean) < 2:
        return False
    if text_clean.isdigit():
        return False
    if conf < min_conf:
        return False
    if re.match(r'^\d+\s*hp', text_lower):
        return False
    if re.match(r'^hp\s*\d+', text_lower):
        return False
    if re.match(r'^\d{1,3}\s*/\s*\d{1,3}', text_clean):
        return False
    if len(text_clean) <= 2 and not text_clean.isalpha():
        return False

    # Reject known non-name words (exact match)
    if text_lower in _NAME_REJECT:
        return False
    if text_lower.rstrip('.') in _NAME_REJECT:
        return False

    # Reject if any individual word matches the reject set
    words = set(text_lower.split())
    if words & _NAME_REJECT:
        return False

    # Reject short single-word candidates that look like garbled banner text
    # (e.g. "Supported", "TRAINEF", "TpA" are OCR misreads of banner words)
    if len(words) == 1 and len(text_clean) <= 12:
        for prefix in _BANNER_PREFIXES:
            if text_lower.startswith(prefix):
                return False

    return True


def _update_candidate(candidates, text_clean, conf, score, source):
    """Add or update a name candidate — keeps the higher-scoring version."""
    key = text_clean.lower()
    new = {
        "name": text_clean,
        "confidence": round(conf * 100, 1),
        "score": round(score, 3),
        "source": source,
    }
    if key not in candidates or score > candidates[key]["score"]:
        candidates[key] = new


# ─────────────────────────────────────────────────────────────
# HP EXTRACTION (NEW)
# ─────────────────────────────────────────────────────────────

def extract_hp(img, reader, ocr_results=None):
    """
    Extract the HP (hit points) value from the top-right of the card.

    HP appears as "XXX HP" or "HP XXX" in large text at the top-right.
    This is a strong cross-referencing signal: Snorlax has 7+ printings
    with different HP values (90, 100, 120, 130, 140, 150, 220, 340),
    so HP alone can narrow 39 candidates to 1-3.

    Can use either:
      - Pre-existing full-image OCR results (avoids redundant OCR call)
      - A dedicated HP ROI crop (if full OCR wasn't run yet)

    Args:
        img: BGR card image (perspective-corrected)
        reader: EasyOCR reader instance
        ocr_results: optional pre-existing OCR results from full-image pass

    Returns:
        (hp_value, confidence) — e.g. (150, 100.0) or (None, 0.0)
    """
    # Strategy 1: Extract from existing full-image OCR results
    # Look for "NNN HP" or "HP NNN" or just "NNN" near top-right
    if ocr_results:
        hp_val, hp_conf = _hp_from_full_ocr(ocr_results, img.shape[:2])
        if hp_val is not None:
            return hp_val, hp_conf

    # Strategy 2: Dedicated HP ROI crop (if full OCR didn't have it)
    h, w = img.shape[:2]
    y0 = int(h * HP_ROI["y_start"])
    y1 = int(h * HP_ROI["y_end"])
    x0 = int(w * HP_ROI["x_start"])
    x1 = int(w * HP_ROI["x_end"])

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        return None, 0.0

    # Upscale
    roi_w = roi.shape[1]
    if roi_w < 300:
        scale = 300 / roi_w
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    try:
        results = reader.readtext(roi)
    except Exception:
        return None, 0.0

    for (bbox, text, conf) in results:
        hp_match = re.search(r'(\d{2,3})\s*hp', text.lower())
        if hp_match:
            hp_val = int(hp_match.group(1))
            if 10 <= hp_val <= 400:  # Reasonable HP range
                logger.info("HP extracted from ROI: %d (conf=%.2f)", hp_val, conf)
                return hp_val, round(conf * 100, 1)

        # "HP NNN" format (less common)
        hp_match2 = re.search(r'hp\s*(\d{2,3})', text.lower())
        if hp_match2:
            hp_val = int(hp_match2.group(1))
            if 10 <= hp_val <= 400:
                logger.info("HP extracted from ROI: %d (conf=%.2f)", hp_val, conf)
                return hp_val, round(conf * 100, 1)

    return None, 0.0


def _hp_from_full_ocr(ocr_results, img_shape):
    """
    Extract HP from pre-existing full-image OCR results.

    Looks for patterns like "150 HP", "HP 150", or standalone 2-3 digit
    numbers in the top 12% of the card with high confidence.

    Returns (hp_int, confidence_pct) or (None, 0.0).
    """
    h, w = img_shape
    hp_candidates = []

    for (bbox, text, conf) in ocr_results:
        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        y_ratio = (min(ys) + max(ys)) / 2 / h
        x_ratio = (min(xs) + max(xs)) / 2 / w

        # HP is in the top portion, typically right side
        if y_ratio > 0.15:
            continue

        text_lower = text.strip().lower()

        # "NNN HP" pattern (most common on modern cards)
        hp_match = re.search(r'(\d{2,3})\s*hp', text_lower)
        if hp_match:
            val = int(hp_match.group(1))
            if 10 <= val <= 400:
                hp_candidates.append((val, conf, x_ratio))
                continue

        # "HP NNN" pattern
        hp_match2 = re.search(r'hp\s*(\d{2,3})', text_lower)
        if hp_match2:
            val = int(hp_match2.group(1))
            if 10 <= val <= 400:
                hp_candidates.append((val, conf, x_ratio))
                continue

        # Standalone number in top-right that could be HP
        # Real-world: OCR often reads "150" without the "HP" suffix
        # Must be: high confidence, right side of card, top region, multiple of 10
        if text.strip().isdigit() and conf > 0.70:
            val = int(text.strip())
            if 30 <= val <= 400 and val % 10 == 0:
                # Prefer right-side readings (HP is always top-right)
                # but accept left-side if it's the only candidate
                position_bonus = 0.9 if x_ratio > 0.55 else 0.7
                hp_candidates.append((val, conf * position_bonus, x_ratio))

    if not hp_candidates:
        return None, 0.0

    # Pick the best: prefer those with explicit "HP" label, then by confidence
    hp_candidates.sort(key=lambda x: x[1], reverse=True)
    best_val, best_conf, _ = hp_candidates[0]
    logger.info("HP extracted from full OCR: %d (conf=%.2f)", best_val, best_conf)
    return best_val, round(best_conf * 100, 1)


# ─────────────────────────────────────────────────────────────
# LEGACY NAME EXTRACTION (preserved as fallback)
# ─────────────────────────────────────────────────────────────

def extract_name(ocr_results):
    """
    LEGACY: Extract card name from full-image OCR results.

    Preserved as a fallback in case the ROI-based approach fails
    (e.g., non-standard card layouts, trainer cards with different
    name placement). The new pipeline calls extract_name_candidates()
    first; this is only called if that returns no results.

    Strategy: Score each OCR result by bounding box area, vertical position,
    and confidence. Filter out known non-name text via skip list.

    Returns (name, confidence_pct) or (None, 0.0).
    """
    if not ocr_results:
        return None, 0.0

    max_y = max(max(pt[1] for pt in bbox) for (bbox, text, conf) in ocr_results)
    if max_y == 0:
        max_y = 1

    candidates = []

    for (bbox, text, conf) in ocr_results:
        text_clean = text.strip()
        text_lower = text_clean.lower()

        # Hard filters
        if any(word in text_lower for word in NAME_SKIP_WORDS):
            continue
        if len(text_clean) < 2 or text_clean.isdigit():
            continue
        if conf < NAME_MIN_CONFIDENCE:
            continue
        if re.match(r'^\d+\s*/\s*\d+', text_clean):
            continue
        if len(text_clean) <= 2 and not text_clean.isalpha():
            continue

        # Bounding box metrics
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        bb_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        y_center = (min(ys) + max(ys)) / 2
        y_ratio = y_center / max_y

        position_score = max(0, 1.0 - (y_ratio * 2.0))
        score = (
            (position_score * NAME_POSITION_WEIGHT)
            + (conf * NAME_CONFIDENCE_WEIGHT)
            + (bb_area * NAME_SIZE_WEIGHT)
        )

        candidates.append((text_clean, conf, score, y_ratio))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[2], reverse=True)

    # Prefer candidates in the top portion of the card
    for (text, conf, score, y_ratio) in candidates:
        if y_ratio < NAME_MAX_Y_RATIO:
            return text, round(conf * 100, 1)

    # Fallback: highest score regardless of position
    best = candidates[0]
    return best[0], round(best[1] * 100, 1)


# ─────────────────────────────────────────────────────────────
# NUMBER EXTRACTION (preserved — consensus scoring)
# ─────────────────────────────────────────────────────────────

def _save_debug_image(img, label, card_type="unknown"):
    """Save a preprocessing step image for debugging."""
    if OCR_DEBUG_DIR is None:
        return
    debug_dir = Path(OCR_DEBUG_DIR)
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000) % 100000
    filename = f"{timestamp}_{card_type}_{label}.png"
    cv2.imwrite(str(debug_dir / filename), img)


def _reduce_highlights(roi_bgr):
    """
    Reduce specular highlights by clamping the V channel in HSV.
    Preserves detail in bright areas that would otherwise saturate to white.
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_clamped = np.clip(v, 0, NUMBER_HIGHLIGHT_CLAMP)
    v_norm = cv2.normalize(v_clamped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv_out = cv2.merge([h, s, v_norm])
    return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)


def _apply_clahe(gray):
    """Apply CLAHE to normalize local contrast (handles uneven phone lighting)."""
    clahe = cv2.createCLAHE(
        clipLimit=NUMBER_CLAHE_CLIP,
        tileGridSize=NUMBER_CLAHE_GRID,
    )
    return clahe.apply(gray)


def _unsharp_mask(gray):
    """Gaussian unsharp mask — sharpens text edges without amplifying noise."""
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=NUMBER_UNSHARP_SIGMA)
    return cv2.addWeighted(
        gray, NUMBER_UNSHARP_WEIGHT,
        blurred, NUMBER_UNSHARP_BLUR_WEIGHT,
        0,
    )


def _preprocess_number_roi(roi_bgr, card_type="pokemon"):
    """
    Full preprocessing pipeline for the card number strip.
    Returns a list of (label, processed_image) variants for OCR.
    """
    variants = []

    # Step 1: Upscale ROI
    scale = NUMBER_UPSCALE_V2
    roi_up = cv2.resize(roi_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _save_debug_image(roi_up, "01_upscaled", card_type)

    # Step 2: Reduce highlights (before grayscale to preserve color info)
    roi_dehighlight = _reduce_highlights(roi_up)
    _save_debug_image(roi_dehighlight, "02_dehighlight", card_type)

    # Step 3: Convert to grayscale (luminance-weighted)
    gray = cv2.cvtColor(roi_dehighlight, cv2.COLOR_BGR2GRAY)
    _save_debug_image(gray, "03_gray", card_type)

    # Step 4: Bilateral filter — denoise while preserving edges
    denoised = cv2.bilateralFilter(
        gray, NUMBER_BILATERAL_D,
        NUMBER_BILATERAL_SIGMA, NUMBER_BILATERAL_SIGMA,
    )
    _save_debug_image(denoised, "04_denoised", card_type)

    # Step 5: CLAHE — equalize local contrast
    clahe_img = _apply_clahe(denoised)
    _save_debug_image(clahe_img, "05_clahe", card_type)

    # Step 6: Unsharp mask — sharpen text edges
    sharpened = _unsharp_mask(clahe_img)
    _save_debug_image(sharpened, "06_sharpened", card_type)

    # Step 7: Generate binarization variants
    # 7a: CLAHE + Otsu (best general purpose)
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _save_debug_image(otsu, "07a_otsu", card_type)
    variants.append(("otsu", otsu))

    # 7b: Adaptive Gaussian
    adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )
    _save_debug_image(adaptive, "07b_adaptive", card_type)
    variants.append(("adaptive", adaptive))

    # 7c: Sharpened grayscale (let EasyOCR handle contrast internally)
    variants.append(("sharpened", sharpened))

    # 7d: Inverted for light-on-dark text (some card eras)
    _, inv = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _save_debug_image(inv, "07d_inverted", card_type)
    variants.append(("inverted", inv))

    return variants


def extract_number(img, reader, card_type="pokemon", set_totals=None):
    """
    Pass 2 (v2): Extract card number from the bottom strip using enhanced
    preprocessing pipeline with CLAHE, highlight reduction, and constrained OCR.

    Args:
        img: BGR card image
        reader: EasyOCR reader instance
        card_type: pokemon/trainer/energy (determines crop region)
        set_totals: dict of total_string → [set_ids] for cross-variant pairing

    Returns (card_number, set_total, confidence_pct) or (None, None, 0.0).
    """
    if set_totals is None:
        set_totals = KNOWN_SET_TOTALS

    h, w = img.shape[:2]

    # Use card-type-specific crop region
    region = NUMBER_REGIONS.get(card_type, NUMBER_REGIONS["pokemon"])
    y_start = int(h * region["y_start"])
    y_end = int(h * region["y_end"])
    x_start = int(w * region["x_start"])
    x_end = int(w * region["x_end"])

    roi = img[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return None, None, 0.0

    logger.info("Number ROI: %dx%d from region y=%.2f–%.2f x=%.2f–%.2f (card_type=%s)",
                roi.shape[1], roi.shape[0],
                region["y_start"], region["y_end"],
                region["x_start"], region["x_end"], card_type)

    _save_debug_image(roi, "00_raw_roi", card_type)

    # Run full preprocessing pipeline
    variants = _preprocess_number_roi(roi, card_type)

    # ── Collect all raw OCR reads across all variants ──
    all_reads = []  # (text, conf, variant_label)

    for label, processed in variants:
        try:
            results = reader.readtext(
                processed,
                allowlist=NUMBER_ALLOWLIST,
                paragraph=False,
                min_size=NUMBER_MIN_SIZE,
                contrast_ths=NUMBER_CONTRAST_THS,
                adjust_contrast=NUMBER_ADJUST_CONTRAST,
            )
        except Exception as e:
            logger.warning("EasyOCR failed on variant %s: %s", label, e)
            continue

        variant_texts = []
        for (bbox, text, conf) in results:
            text = text.strip()
            if text:
                all_reads.append((text, conf, label))
                variant_texts.append(f"'{text}' c={conf:.2f}")

        logger.info("  [%s] OCR reads: %s", label, " | ".join(variant_texts) if variant_texts else "(none)")

    # ── Also run a free-text pass (no allowlist) for additional data ──
    try:
        free_results = reader.readtext(
            variants[2][1] if len(variants) > 2 else roi,  # use sharpened variant
            paragraph=False,
            min_size=NUMBER_MIN_SIZE,
        )
        free_texts = []
        for (bbox, text, conf) in free_results:
            text = text.strip()
            if text and re.search(r'\d', text):
                all_reads.append((text, conf, "free-text"))
                free_texts.append(f"'{text}' c={conf:.2f}")
        if free_texts:
            logger.info("  [free-text] OCR reads: %s", " | ".join(free_texts))
    except Exception:
        pass

    # ── Extract number/total candidates using 3 strategies ──
    candidates = {}  # (number, total) → {count, best_conf, sources}

    for text, conf, variant in all_reads:
        # Strategy 1: Explicit separator (057/203)
        sep_match = re.search(r'(\d{1,3})\s*[/|]\s*(\d{1,3})', text)
        if sep_match:
            num = sep_match.group(1).lstrip("0") or "0"
            total = sep_match.group(2).lstrip("0") or "0"
            _add_candidate(candidates, num, total, conf, variant)
            continue

        # Strategy 2: Concat-digit splitting against known set totals
        # e.g., "055078" → try splitting at each position, check if right
        # portion matches a known set total
        digits = re.sub(r'[^0-9]', '', text)
        if len(digits) >= 4 and set_totals:
            for split_pos in range(1, len(digits)):
                left = digits[:split_pos].lstrip("0") or "0"
                right = digits[split_pos:].lstrip("0") or "0"
                if right in set_totals:
                    _add_candidate(candidates, left, right, conf, variant)

    # Strategy 3: Cross-variant pairing
    # If variant A reads "055" (standalone number) and variant B reads "078"
    # as part of a longer string containing a known total, combine them.
    standalone_numbers = []
    known_totals_seen = []

    for text, conf, variant in all_reads:
        digits = re.sub(r'[^0-9]', '', text)
        if not digits:
            continue

        # Standalone short number (likely the card number alone)
        if len(digits) <= 3 and conf > 0.20:
            num = digits.lstrip("0") or "0"
            standalone_numbers.append((num, conf, variant))

        # Check if this read contains a known set total
        if set_totals:
            for total_str in set_totals:
                if total_str in digits and len(total_str) >= 2:
                    known_totals_seen.append((total_str, conf, variant))

    # Pair standalone numbers with known totals from different variants
    for num, num_conf, num_var in standalone_numbers:
        for total, total_conf, total_var in known_totals_seen:
            if num_var != total_var:  # Cross-variant pairing
                combined_conf = min(num_conf, total_conf) * 0.95
                _add_candidate(candidates, num, total, combined_conf,
                               f"{num_var}+{total_var}")

    # ── Score and rank candidates ──
    if not candidates:
        logger.info("Number extraction: no candidates found")
        return None, None, 0.0

    scored = []
    for (num, total), info in candidates.items():
        # Score: count × 0.4 + best_conf × 0.6 (consensus weighting)
        score = (info["count"] * 0.4) + (info["best_conf"] * 0.6)
        scored.append((num, total, score, info["count"], info["best_conf"]))

    scored.sort(key=lambda x: x[2], reverse=True)

    for num, total, score, count, conf in scored[:5]:
        logger.info("  Candidate: %s/%s count=%d conf=%.2f score=%.2f",
                     num, total, count, conf, score)

    winner = scored[0]
    logger.info("Number extraction: %s/%s (score=%.2f, count=%d, conf=%.2f)",
                winner[0], winner[1], winner[2], winner[3], winner[4])

    return winner[0], winner[1], round(winner[4] * 100, 1)


def _add_candidate(candidates, num, total, conf, source):
    """Add or update a number/total candidate in the candidates dict."""
    key = (num, total)
    if key not in candidates:
        candidates[key] = {"count": 0, "best_conf": 0.0, "sources": []}
    candidates[key]["count"] += 1
    candidates[key]["best_conf"] = max(candidates[key]["best_conf"], conf)
    candidates[key]["sources"].append(source)

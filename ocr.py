"""
ocr.py
Card text extraction via EasyOCR.
  - detect_card_type: classify as Pokemon, Trainer, or Energy
  - extract_name: identify card name from full-image OCR
  - extract_number: extract card number from bottom strip
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


def extract_name(ocr_results):
    """
    Pass 1: Extract card name from full-image OCR results.

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

    # OCR_DEBUG: log all candidates with position
    _max_y_dbg = max(max(pt[1] for pt in bbox) for (bbox, text, conf) in ocr_results) if ocr_results else 1
    print("    [OCR DEBUG] All detected text:")
    for (bbox, text, conf) in ocr_results:
        ys = [pt[1] for pt in bbox]
        y_pct = round(((min(ys) + max(ys)) / 2) / _max_y_dbg * 100)
        print(f"      y={y_pct:3d}%  conf={conf:.2f}  text='{text}'")
    print("    [OCR DEBUG] ---")

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


def extract_number(img, reader, card_type="pokemon"):
    """
    Pass 2 (v2): Extract card number from the bottom strip using enhanced
    preprocessing pipeline with CLAHE, highlight reduction, and constrained OCR.

    Returns (card_number, set_total, confidence_pct) or (None, None, 0.0).
    """
    h, w = img.shape[:2]

    # Use card-type-specific crop region (tighter than v1)
    region = NUMBER_REGIONS.get(card_type, NUMBER_REGIONS["pokemon"])
    y_start = int(h * region["y_start"])
    y_end = int(h * region["y_end"])
    x_start = int(w * region["x_start"])
    x_end = int(w * region["x_end"])

    roi = img[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        return None, None, 0.0

    _save_debug_image(roi, "00_raw_roi", card_type)

    # Run full preprocessing pipeline
    variants = _preprocess_number_roi(roi, card_type)

    # Expanded regex patterns to catch common OCR misreads of "/"
    number_patterns = [
        re.compile(r'(\d{1,3})\s*/\s*(\d{1,3})'),            # Standard: 057/203
        re.compile(r'(\d{1,3})\s*[|1lI)\]7\\]\s*(\d{1,3})'),  # OCR misreads of /
        re.compile(r'(\d{1,3})\s*[.,]\s*(\d{1,3})'),           # Dot/comma misread
    ]

    best_match = (None, None, 0.0, "")  # (number, total, confidence, variant_label)

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

        for (bbox, text, conf) in results:
            for pattern in number_patterns:
                match = pattern.search(text)
                if match and conf > best_match[2]:
                    candidate_num = match.group(1).lstrip("0") or "0"
                    candidate_total = match.group(2)
                    best_match = (candidate_num, candidate_total, conf, label)

                    # Early exit on high confidence
                    if conf > NUMBER_EARLY_EXIT_CONF:
                        logger.info(
                            "Number extraction: %s/%s (conf=%.2f, variant=%s)",
                            candidate_num, candidate_total, conf, label,
                        )
                        return best_match[0], best_match[1], round(best_match[2] * 100, 1)

    if best_match[0]:
        logger.info(
            "Number extraction: %s/%s (conf=%.2f, variant=%s)",
            best_match[0], best_match[1], best_match[2], best_match[3],
        )

    return best_match[0], best_match[1], round(best_match[2] * 100, 1)

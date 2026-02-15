"""
stamp.py
1st Edition stamp detection via multi-scale template matching.
Includes bilateral filter for holo suppression and contrast validation.
"""

import os
import cv2
import numpy as np

from config import (
    TEMPLATE_PATH,
    STAMP_THRESHOLD, STAMP_CONTRAST_MIN, STAMP_CONTRAST_PENALTY,
    STAMP_SCALE_MIN, STAMP_SCALE_MAX, STAMP_SCALE_STEPS,
    BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE,
)


def load_stamp_template():
    """Load and prepare the 1st Edition stamp template (grayscale)."""
    if not os.path.exists(TEMPLATE_PATH):
        return None
    template = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    return template


def check_stamp(card_img, card_type, template):
    """
    Search for 1st Edition stamp via template matching.

    Stamp locations by card type (verified from actual cards):
      - Pokemon: mid-left, between art and Pokedex bar
      - Trainer: bottom-left, above copyright line
      - Energy:  top-right corner, near ENERGY banner

    Anti-false-positive measures:
      - Bilateral filter smooths holographic gradients while preserving stamp edges
      - Contrast validation rejects low-contrast matches (holo noise)

    Returns (is_1st_edition, confidence_pct).
    """
    if template is None:
        return False, 0.0

    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Search regions by card type: (y_start%, y_end%, x_start%, x_end%)
    if card_type == "energy":
        regions = [
            (0.01, 0.18, 0.58, 0.99),
            (0.01, 0.22, 0.50, 0.99),
        ]
    elif card_type == "trainer":
        regions = [
            (0.75, 0.93, 0.02, 0.22),
            (0.70, 0.95, 0.02, 0.28),
        ]
    else:  # pokemon
        regions = [
            (0.44, 0.58, 0.02, 0.16),
            (0.40, 0.62, 0.01, 0.20),
            (0.48, 0.56, 0.03, 0.13),
        ]

    best_score = 0.0
    best_loc = None
    best_roi = None
    best_size = (0, 0)
    th, tw = template.shape

    for (y1p, y2p, x1p, x2p) in regions:
        roi = gray[int(h * y1p):int(h * y2p), int(w * x1p):int(w * x2p)]
        if roi.size == 0:
            continue

        roi_filtered = cv2.bilateralFilter(
            roi, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
        )

        for scale in np.linspace(STAMP_SCALE_MIN, STAMP_SCALE_MAX, STAMP_SCALE_STEPS):
            new_w = int(tw * scale)
            new_h = int(th * scale)

            if new_h >= roi_filtered.shape[0] or new_w >= roi_filtered.shape[1]:
                continue
            if new_w < 8 or new_h < 8:
                continue

            resized = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.matchTemplate(roi_filtered, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_score:
                best_score = max_val
                best_loc = max_loc
                best_roi = roi  # unfiltered — for contrast check
                best_size = (new_w, new_h)

    # Contrast validation: reject low-contrast matches (holo noise)
    if best_score > 0.40 and best_roi is not None and best_loc is not None:
        bw, bh = best_size
        x, y = best_loc
        patch = best_roi[y:y + bh, x:x + bw]
        if patch.size > 0:
            stddev = np.std(patch)
            if stddev < STAMP_CONTRAST_MIN:
                best_score *= STAMP_CONTRAST_PENALTY

    is_1st = best_score > STAMP_THRESHOLD
    return is_1st, round(best_score * 100, 1)

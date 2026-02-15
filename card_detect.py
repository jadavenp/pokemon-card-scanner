"""
card_detect.py
Detect and extract a card from a camera photo using OpenCV.

Takes a raw image (from phone camera), finds the card contour,
applies perspective correction, and returns a clean, upright card image
suitable for OCR and hash matching.

Based on the detect_card() prototype from live_scanner.py.
"""

import cv2
import numpy as np
import logging

from config import CARD_MIN_OUTPUT_WIDTH

logger = logging.getLogger("card_detect")


def detect_and_crop_card(img, min_area_ratio=0.05, max_area_ratio=0.95):
    """
    Detect a card in the image and return a perspective-corrected crop.

    The input image is expected to be a phone photo that's already roughly
    framed around a card (from the mobile UI guide rectangle), but may
    include some background, and the card may be slightly tilted.

    Args:
        img: BGR image (numpy array from cv2.imread)
        min_area_ratio: Minimum card area as fraction of image area
        max_area_ratio: Maximum card area as fraction of image area

    Returns:
        (warped_card, success): Tuple of (perspective-corrected card image, True)
        or (original_img, False) if no card detected.
    """
    if img is None or img.size == 0:
        return img, False

    h, w = img.shape[:2]
    total_area = h * w

    # ── Preprocessing ──
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 40, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # ── Find contours ──
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        area_ratio = area / total_area

        # Card should be a significant portion of the frame
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # Try to approximate to a 4-sided polygon
        peri = cv2.arcLength(cnt, True)
        approx = None
        for eps in [0.02, 0.03, 0.04, 0.05]:
            candidate = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(candidate) == 4:
                approx = candidate
                break

        # If we can't get exactly 4 vertices but the contour passes
        # area/aspect/solidity checks, fall back to minAreaRect corners
        use_box_fallback = (approx is None)
        if approx is None:
            # Only allow fallback if contour is large enough and roughly card-shaped
            if area_ratio < 0.10:
                continue

        # Check solidity (card should fill its bounding rect well)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue

        rect_area = rw * rh
        solidity = area / rect_area if rect_area > 0 else 0
        if solidity < 0.75:
            continue

        # Check card aspect ratio (standard card is ~0.714 = 2.5/3.5)
        aspect = min(rw, rh) / max(rw, rh)
        if not (0.55 < aspect < 0.85):
            continue

        # ── Perspective correction ──
        if approx is not None:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            pts = box.astype(np.float32)
        sorted_pts = sorted(pts, key=lambda p: p[1])
        top = sorted(sorted_pts[:2], key=lambda p: p[0])
        bottom = sorted(sorted_pts[2:], key=lambda p: p[0])
        ordered = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)

        # Output at standard card proportions
        card_w = int(max(rw, rh))
        card_h = int(card_w * 3.5 / 2.5)
        if rw > rh:
            card_w, card_h = card_h, card_w

        # Ensure minimum output size for OCR readability
        # (480px minimum ensures bottom-strip text is ~48-67px tall)
        if card_w < CARD_MIN_OUTPUT_WIDTH:
            scale = CARD_MIN_OUTPUT_WIDTH / card_w
            card_w = CARD_MIN_OUTPUT_WIDTH
            card_h = int(card_h * scale)

        dst = np.array([
            [0, 0],
            [card_w - 1, 0],
            [card_w - 1, card_h - 1],
            [0, card_h - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(img, M, (card_w, card_h))

        logger.info(
            "Card detected: %dx%d, aspect=%.3f, solidity=%.2f, area=%.1f%%",
            card_w, card_h, aspect, solidity, area_ratio * 100
        )
        return warped, True

    logger.info("No card detected in frame — using original image")
    return img, False

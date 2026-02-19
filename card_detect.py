"""
card_detect.py — Detect and extract a trading card from a camera photo.

Takes a raw image (from phone camera or webcam), finds the card contour,
applies perspective correction, and returns a clean, upright card image
suitable for OCR and hash matching.

Pipeline:
    1. Grayscale conversion (no blur — iPhone computational photography
       already denoises; blur destroys edge sharpness for no benefit)
    2. Auto-Canny edge detection with adaptive thresholds derived from
       image median (sigma passes: tight → standard → permissive)
    3. Contour filtering by area, solidity, and aspect ratio
    4. 4-corner extraction (approxPolyDP preferred, minAreaRect fallback)
    5. Perspective warp to standard card proportions (2.5:3.5 portrait)
    6. Upscale to minimum width for OCR readability

Design notes:
    - Auto-Canny (Rosebrock 2015): thresholds computed as median ± sigma%,
      adapting to each image's brightness. Replaces fixed thresholds that
      failed on holo reflections and varying lighting conditions.
    - No Gaussian blur: modern phone cameras (iPhone 12+) apply multi-frame
      noise reduction and computational HDR before the image reaches the app.
      Blurring a denoised 12MP image only softens edges.
    - Multi-pass sigma approach: tight sigma (0.33) catches clean high-contrast
      shots; wider sigma (0.50, 0.67) catches holo/low-contrast cards.
    - The solidity check (contour area / bounding rect area >= 0.75) rejects
      fragmented contours from internal card features (art, text boxes).
    - Corner ordering uses y-sort then x-sort to get [TL, TR, BR, BL] for the
      perspective transform. This is invariant to card rotation up to ~45 deg.
    - Output dimensions use min(rw, rh) as width because minAreaRect's
      width/height assignment depends on rotation angle, not image axes.
"""

import cv2
import numpy as np
import logging

from config import CARD_MIN_OUTPUT_WIDTH

logger = logging.getLogger("card_detect")

# Standard card proportions
CARD_ASPECT = 2.5 / 3.5  # width / height = 0.714

# Auto-Canny sigma values: controls threshold band around image median.
# sigma=0.33 → thresholds at median ± 33% (tight, catches clean shots)
# sigma=0.50 → thresholds at median ± 50% (catches holo/textured cards)
# sigma=0.67 → thresholds at median ± 67% (permissive, low-contrast scenarios)
_SIGMA_PASSES = [0.33, 0.50, 0.67]


def detect_and_crop_card(img, min_area_ratio=0.05, max_area_ratio=0.995):
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
        (warped_card, True) on success — perspective-corrected portrait image.
        (original_img, False) if no card detected.
    """
    if img is None or img.size == 0:
        return img, False

    h, w = img.shape[:2]
    total_area = h * w

    # Preprocessing: grayscale only — no blur.
    # iPhone computational photography already denoises; blurring a clean
    # 12MP image only softens the card-to-background edge.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try each auto-Canny sigma pass — stop on first successful detection
    for sigma in _SIGMA_PASSES:
        # Auto-Canny: derive thresholds from image median
        v = np.median(gray)
        canny_lo = int(max(0,   (1.0 - sigma) * v))
        canny_hi = int(min(255, (1.0 + sigma) * v))

        result = _try_detect(
            img, gray, canny_lo, canny_hi,
            total_area, min_area_ratio, max_area_ratio
        )
        if result is not None:
            warped, card_w, card_h, aspect, solidity, area_ratio = result
            logger.info(
                "Card detected: %dx%d, aspect=%.3f, solidity=%.2f, "
                "area=%.1f%%, auto-canny sigma=%.2f thresholds=(%d,%d)",
                card_w, card_h, aspect, solidity,
                area_ratio * 100, sigma, canny_lo, canny_hi
            )
            return warped, True

    logger.info("No card detected in frame — using original image")
    return img, False


def _try_detect(img, gray, canny_lo, canny_hi,
                total_area, min_area_ratio, max_area_ratio):
    """
    Single detection pass with given Canny thresholds.

    Args:
        img: Original BGR image (for perspective warp output)
        gray: Grayscale image (no blur applied — raw from phone camera)
        canny_lo: Lower Canny threshold (auto-computed from median)
        canny_hi: Upper Canny threshold (auto-computed from median)

    Returns (warped, card_w, card_h, aspect, solidity, area_ratio)
    on success, or None on failure.
    """
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:10]:
        area = cv2.contourArea(cnt)
        area_ratio = area / total_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # ── Shape validation ──
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue

        # Solidity: card should fill its bounding rectangle well.
        # Fragmented contours (from internal card art) score low here.
        solidity = area / (rw * rh)
        if solidity < 0.75:
            continue

        # Aspect ratio: standard card is 2.5" x 3.5" = 0.714
        aspect = min(rw, rh) / max(rw, rh)
        if not (0.55 < aspect < 0.85):
            continue

        # ── Extract 4 corners ──
        corners = _extract_corners(cnt, rect, area_ratio)
        if corners is None:
            continue

        # ── Perspective warp to portrait ──
        warped, card_w, card_h = _warp_to_portrait(img, corners, rw, rh)
        return warped, card_w, card_h, aspect, solidity, area_ratio

    return None


def _extract_corners(cnt, rect, area_ratio):
    """
    Get 4 ordered corners [TL, TR, BR, BL] from a contour.

    Prefers approxPolyDP on convex hull (actual card corners) over
    minAreaRect box (may not match true corners on tilted cards).

    Convex hull step eliminates concavities from shadows and edge noise
    before polygon approximation, so approxPolyDP sees a clean outline
    rather than a blob with inward dents.

    Returns np.float32 array of shape (4, 2) or None.
    """
    # Compute convex hull first to eliminate shadow/noise concavities
    hull = cv2.convexHull(cnt)
    hull_peri = cv2.arcLength(hull, True)

    # Try progressively looser approximation on the hull to get exactly 4 vertices
    approx = None
    for eps in [0.02, 0.03, 0.04, 0.05]:
        candidate = cv2.approxPolyDP(hull, eps * hull_peri, True)
        if len(candidate) == 4:
            approx = candidate
            break

    if approx is not None:
        pts = approx.reshape(4, 2).astype(np.float32)
    elif area_ratio >= 0.10:
        # Fallback: use minAreaRect box corners for large contours
        # that couldn't be approximated to 4 vertices
        box = cv2.boxPoints(rect)
        pts = box.astype(np.float32)
    else:
        return None

    return _order_corners(pts)


def _order_corners(pts):
    """
    Order 4 points as [TL, TR, BR, BL] for perspective transform.

    Method: sum/difference of coordinates — rotation-invariant up to 90 deg.
      TL = smallest (x+y)   BR = largest (x+y)
      TR = smallest (y-x)   BL = largest (y-x)

    Replaces the previous y-sort + x-sort approach, which broke when the
    card was rotated more than ~15° (approxPolyDP corners are not
    axis-aligned, so top-2/bottom-2 by y-value mis-assigns corners).
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR: largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR: smallest y-x
    rect[3] = pts[np.argmax(d)]   # BL: largest y-x
    return rect


def _warp_to_portrait(img, corners, rw, rh):
    """
    Perspective-warp the card region to a clean portrait rectangle.

    Output width = min(rw, rh) because minAreaRect's width/height
    assignment is rotation-dependent, not axis-aligned. The short edge
    is always the card width for a portrait card.

    Enforces minimum width (CARD_MIN_OUTPUT_WIDTH) for OCR readability.
    """
    card_w = int(min(rw, rh))
    card_h = int(card_w * 3.5 / 2.5)

    # Upscale small cards for OCR readability
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

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (card_w, card_h))
    return warped, card_w, card_h

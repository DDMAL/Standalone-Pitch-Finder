"""
Shared debug-overlay drawing helpers.

Glyph bboxes on these manuscript scans are ~20-30px, too small to caption
legibly at native resolution. load_scaled_image upscales the canvas first
so labels get enough real pixels to be readable without shrinking relative
to the glyphs; draw_labeled_box then draws a box plus a white-backed label
so text doesn't get lost against parchment texture / dark ink.
"""

from pathlib import Path

import cv2


def load_scaled_image(image_path: Path, scale: float):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def draw_labeled_box(img, ulx: float, uly: float, ncols: float, nrows: float,
                      label: str, color: tuple, scale: float):
    p1 = (round(ulx * scale), round(uly * scale))
    p2 = (round((ulx + ncols) * scale), round((uly + nrows) * scale))

    box_thickness = max(1, round(scale))
    cv2.rectangle(img, p1, p2, color, box_thickness)

    if not label:
        return

    font_scale = 0.45 * scale
    font_thickness = max(1, round(scale * 0.7))
    text_org = (p1[0], max(0, p1[1] - round(4 * scale)))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    bg_p1 = (text_org[0] - 1, text_org[1] - th - baseline)
    bg_p2 = (text_org[0] + tw + 1, text_org[1] + baseline)
    cv2.rectangle(img, bg_p1, bg_p2, (255, 255, 255), -1)
    cv2.putText(img, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                font_thickness, cv2.LINE_AA)

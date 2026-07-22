"""
Pixel-level glyph analysis for the Rodan-style pitch finder.

Ports the pixel-analysis half of Rodan's heuristic pitch finder --
_x_projection_vector / get_subimage / _center_of_mass / _vector_process_f_clef
in https://github.com/DDMAL/Rodan/blob/master/rodan-main/code/rodan/jobs/heuristic_pitch_finding/PitchFinding.py
-- without needing Gamera: we crop straight from our own manuscript image
using the IC bbox, instead of decoding Gamera's onebit image encoding.

Rodan's rationale (kept here): a glyph's bounding box is not a reliable
pitch reference on its own -- a virga's descending stem, or a podatus's
second notehead, pull the bbox's geometric center away from the actual
notehead. So for a handful of classes, only a sub-region of the bbox is
used for the row-projection centroid:
  - virga: top region only (the stem trails below the notehead)
  - podatus2b/3/4/5, scandicus22b: bottom-left region only (these are
    drawn ligated with the reference notehead at the bottom-left)
  - clef.f* (any F-clef variant): right half only, trimmed to its own ink
    extent (the two dots on an F-clef's left side aren't the reference point)
  - everything else: full height, width capped at the average punctum size
    (so an unusually wide multi-note glyph doesn't get diluted by grabbing
    a neighboring symbol)
"""

import numpy as np
import cv2

from ic_io import Glyph

BOTTOM_LEFT_CLASSES = {"neume.podatus2b", "neume.podatus3", "neume.podatus4",
                        "neume.podatus5", "neume.scandicus22b"}


def average_punctum(glyphs: list[Glyph]) -> float:
    """Average bbox width of punctum/virga glyphs -- Rodan's proxy for
    'one notehead's width', used to size the special-case crops below."""
    widths = [g.ncols for g in glyphs if g.class_name in ("neume.punctum", "neume.virga")]
    return sum(widths) / len(widths) if widths else 0.0


def crop_and_binarize(image: np.ndarray, ulx: float, uly: float, ncols: float, nrows: float) -> np.ndarray:
    """Crop a region from the page image and Otsu-binarize it (ink = 255)."""
    ulx, uly, ncols, nrows = round(ulx), round(uly), round(ncols), round(nrows)
    h, w = image.shape[:2]
    x0, y0 = max(0, ulx), max(0, uly)
    x1, y1 = min(w, ulx + ncols), min(h, uly + nrows)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=np.uint8)

    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def row_projection_centroid(binary_crop: np.ndarray) -> float:
    """Ink-weighted row centroid, 1-indexed (row 1 = the crop's top row).

    Matches Rodan's _center_of_mass exactly: s = sum((i+1)*count_i),
    v = sum(count_i), com = s/v (0 if there's no ink at all).
    """
    if binary_crop.size == 0:
        return 0.0
    projection = (binary_crop != 0).sum(axis=1).astype(float)
    total = projection.sum()
    if total == 0:
        return 0.0
    weights = np.arange(1, len(projection) + 1, dtype=float)
    return float((weights * projection).sum() / total)


def _f_clef_right_region(image: np.ndarray, glyph: Glyph) -> tuple:
    """Approximates _vector_process_f_clef: use the right half of the bbox,
    trimmed to its own ink extent, instead of Gamera's connected-component
    union (which needs real Gamera objects we don't have).

    Returns (ulx, uly, ncols, nrows, y_add) for the trimmed right-half region.
    y_add is the region's top offset relative to the original glyph top.
    """
    right_ulx = glyph.ulx + glyph.ncols // 2
    right_ncols = glyph.ulx + glyph.ncols - right_ulx
    crop = crop_and_binarize(image, right_ulx, glyph.uly, right_ncols, glyph.nrows)
    rows_with_ink = np.where((crop != 0).any(axis=1))[0]
    if len(rows_with_ink) == 0:
        return right_ulx, glyph.uly, right_ncols, glyph.nrows, 0.0
    top, bottom = int(rows_with_ink[0]), int(rows_with_ink[-1])
    return right_ulx, glyph.uly + top, right_ncols, bottom - top + 1, float(top)


def reference_row(image: np.ndarray, glyph: Glyph, avg_punctum: float,
                   discard_size: int = 12, subimage_width_factor: float = 0.8) -> float:
    """The pitch-reference row, as an offset from the glyph's own top edge
    (uly). Add to glyph.uly to get a page-pixel y coordinate.

    Tiny glyphs (both dims <= discard_size) skip pixel analysis entirely,
    matching Rodan's center_of_mass=0 fallback for glyphs too small to
    meaningfully binarize/project.
    """
    if glyph.ncols <= discard_size and glyph.nrows <= discard_size:
        return 0.0

    ulx, uly, ncols, nrows = glyph.ulx, glyph.uly, glyph.ncols, glyph.nrows
    y_add = 0.0

    if glyph.class_name.startswith("clef.f"):
        ulx, uly, ncols, nrows, y_add = _f_clef_right_region(image, glyph)

    extend_cols = ncols if ncols < avg_punctum else avg_punctum * subimage_width_factor
    extend_rows = nrows if nrows < avg_punctum else avg_punctum
    extend_cols = max(1, round(extend_cols))
    extend_rows = max(1, round(extend_rows))

    if glyph.class_name in BOTTOM_LEFT_CLASSES:
        crop_uly = uly + nrows - extend_rows
        crop = crop_and_binarize(image, ulx, crop_uly, extend_cols, extend_rows)
        return y_add + (nrows - extend_rows) + row_projection_centroid(crop)

    if glyph.class_name == "neume.virga":
        crop = crop_and_binarize(image, ulx, uly, extend_cols, extend_rows)
        return y_add + row_projection_centroid(crop)

    # Default: full height, width capped to avoid grabbing a neighbor.
    crop = crop_and_binarize(image, ulx, uly, extend_cols, nrows)
    return y_add + row_projection_centroid(crop)

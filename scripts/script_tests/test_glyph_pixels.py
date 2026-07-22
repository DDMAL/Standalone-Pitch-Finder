"""Sanity checks for pixel-level glyph analysis (glyph_pixels.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_io import Glyph
from glyph_pixels import (
    average_punctum, crop_and_binarize, row_projection_centroid, reference_row,
)

BACKGROUND = 220  # light parchment
INK = 20          # dark ink


def make_glyph(index, ulx, uly, ncols, nrows, class_name, state="AUTOMATIC", confidence=0.9):
    return Glyph(index=index, ulx=ulx, uly=uly, nrows=nrows, ncols=ncols,
                 class_name=class_name, confidence=confidence, state=state)


def blank_page(width=200, height=200):
    return np.full((height, width), BACKGROUND, dtype=np.uint8)


def test_average_punctum():
    glyphs = [
        make_glyph(0, 0, 0, 10, 10, "neume.punctum"),
        make_glyph(1, 0, 0, 20, 10, "neume.virga"),
        make_glyph(2, 0, 0, 999, 10, "neume.clivis2"),  # not counted
    ]
    assert average_punctum(glyphs) == 15.0


def test_average_punctum_empty():
    assert average_punctum([]) == 0.0


def test_crop_and_binarize_marks_ink_as_foreground():
    # Otsu needs some contrast to find a threshold, so crop a region that's
    # part ink, part background rather than pure uniform ink.
    img = blank_page()
    img[10:20, 10:15] = INK
    crop = crop_and_binarize(img, 10, 10, 10, 10)
    assert crop.shape == (10, 10)
    assert (crop[:, :5] != 0).all()   # ink half -> foreground
    assert (crop[:, 5:] == 0).all()   # background half -> background


def test_crop_and_binarize_out_of_bounds_returns_empty():
    img = blank_page(50, 50)
    crop = crop_and_binarize(img, 1000, 1000, 10, 10)
    assert crop.size == 0


def test_row_projection_centroid_weighted_mean():
    # 4 rows, all-ink in row 0 only (10 px) -> centroid should be row 1 (1-indexed).
    binary = np.zeros((4, 10), dtype=np.uint8)
    binary[0, :] = 255
    assert row_projection_centroid(binary) == 1.0

    # Ink split evenly between row 0 and row 3 -> centroid = (1*10 + 4*10)/20 = 2.5
    binary2 = np.zeros((4, 10), dtype=np.uint8)
    binary2[0, :] = 255
    binary2[3, :] = 255
    assert row_projection_centroid(binary2) == 2.5


def test_row_projection_centroid_no_ink_is_zero():
    binary = np.zeros((5, 5), dtype=np.uint8)
    assert row_projection_centroid(binary) == 0.0


def test_reference_row_skips_tiny_glyphs():
    img = blank_page()
    g = make_glyph(0, 5, 5, 8, 8, "neume.punctum")  # both dims <= discard_size(12)
    assert reference_row(img, g, avg_punctum=15.0, discard_size=12) == 0.0


def test_virga_reference_excludes_the_stem():
    # Virga: a dense notehead block in the top ~avg_punctum rows, then a
    # thin 1px-wide stem trailing far below. The reference row should land
    # near the notehead, not dragged down toward the bbox's geometric center
    # by the long stem.
    img = blank_page()
    ulx, uly, ncols, nrows = 20, 20, 12, 80
    avg_punctum = 12.0
    img[uly:uly + 12, ulx:ulx + ncols] = INK               # notehead block, rows 0-11
    img[uly + 12:uly + nrows, ulx + 5:ulx + 7] = INK        # thin stem, rows 12-79

    g = make_glyph(0, ulx, uly, ncols, nrows, "neume.virga")
    offset = reference_row(img, g, avg_punctum=avg_punctum, discard_size=12)

    bbox_center = nrows / 2  # naive full-bbox centroid would be way down here (~40)
    assert offset < bbox_center
    assert offset < 12  # centroid should stay within the notehead block itself


def test_podatus_bottom_left_reference_excludes_top_right_ink():
    # podatus2b/3/etc: only the bottom-left corner should feed the centroid.
    # Put dense ink in the top-right (should be ignored) and a smaller block
    # bottom-left (should be picked up).
    img = blank_page()
    ulx, uly, ncols, nrows = 20, 20, 30, 40
    avg_punctum = 12.0
    img[uly:uly + 12, ulx + 15:ulx + 30] = INK          # top-right block, rows 0-11
    img[uly + 28:uly + 40, ulx:ulx + 10] = INK           # bottom-left block, rows 28-39

    g = make_glyph(0, ulx, uly, ncols, nrows, "neume.podatus3")
    offset = reference_row(img, g, avg_punctum=avg_punctum, discard_size=12)

    # Offset is relative to glyph top; bottom-left block spans rows 28-39.
    assert 24 <= offset <= 40


def test_f_clef_uses_right_half_only():
    # Two dots on the left (should be excluded), a solid block on the right
    # (should be the reference). class_name uses the generalized clef.f*
    # matching (e.g. clef.f2).
    img = blank_page()
    ulx, uly, ncols, nrows = 20, 20, 30, 30
    img[uly + 5:uly + 8, ulx:ulx + 4] = INK        # left dot 1
    img[uly + 15:uly + 18, ulx:ulx + 4] = INK      # left dot 2
    img[uly + 10:uly + 20, ulx + 20:ulx + 30] = INK  # right-side ink block

    g = make_glyph(0, ulx, uly, ncols, nrows, "clef.f2")
    offset = reference_row(img, g, avg_punctum=12.0, discard_size=12)

    # Should land within the right-side block's row range (10-19), not the dots.
    assert 8 <= offset <= 22

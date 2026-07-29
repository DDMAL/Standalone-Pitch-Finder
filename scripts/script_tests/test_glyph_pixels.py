"""Sanity checks for pixel-level glyph analysis (glyph_pixels.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_io import Glyph
from glyph_pixels import (
    average_punctum, crop_and_binarize, row_projection_centroid, reference_row,
    reference_point, reference_region,
    REGION_FULL, REGION_TOP, REGION_BOTTOM_LEFT, REGION_F_CLEF_RIGHT,
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
    #
    # The notehead is drawn narrower than the bbox on purpose: Otsu needs both
    # ink and background inside the crop to find a threshold, and a crop that
    # is uniformly ink binarizes to *no* ink at all (centroid 0.0), which would
    # satisfy the "not dragged down" assertions without measuring anything.
    img = blank_page()
    ulx, uly, ncols, nrows = 20, 20, 12, 80
    avg_punctum = 12.0
    img[uly:uly + 12, ulx + 1:ulx + 9] = INK                # notehead block, rows 0-11
    img[uly + 12:uly + nrows, ulx + 5:ulx + 7] = INK        # thin stem, rows 12-79

    g = make_glyph(0, ulx, uly, ncols, nrows, "neume.virga")
    offset = reference_row(img, g, avg_punctum=avg_punctum, discard_size=12)

    # Same bbox, but a class with no special case -- the full-height crop that
    # the stem does drag down. That is the value the virga rule exists to avoid.
    naive = reference_row(img, make_glyph(1, ulx, uly, ncols, nrows, "neume.punctum"),
                          avg_punctum=avg_punctum, discard_size=12)

    print(f"virga offset {offset:.2f} vs full-height offset {naive:.2f}")
    assert 0 < offset <= 12          # inside the notehead block, and measured
    assert offset < naive - 10       # decisively above the stem-biased value


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


def test_reference_region_labels_which_crop_rule_fired():
    """The region label is what tells a decomposing caller which notehead the
    centroid is (see pitch_finder._anchor_interval), so each crop rule has to
    report itself."""
    img = blank_page()
    cases = [
        ("neume.punctum", REGION_FULL),
        ("neume.virga", REGION_TOP),
        ("neume.podatus3", REGION_BOTTOM_LEFT),
        ("neume.scandicus22b", REGION_BOTTOM_LEFT),
        ("clef.f2", REGION_F_CLEF_RIGHT),
        ("neume.clivis2", REGION_FULL),
    ]
    for class_name, expected in cases:
        g = make_glyph(0, 20, 20, 30, 40, class_name)
        region = reference_region(img, g, avg_punctum=12.0)
        assert region is not None
        assert region.region == expected, f"{class_name} -> {region.region}"


def test_reference_point_x_is_the_crops_own_center():
    """The centroid y was measured from the crop's columns, so x has to be that
    band's center -- not the bbox center (which for a wide ligature is a column
    the measurement never looked at)."""
    img = blank_page()
    ulx, uly, ncols, nrows = 20, 20, 40, 30
    img[uly + 5:uly + 25, ulx + 1:ulx + 9] = INK

    g = make_glyph(0, ulx, uly, ncols, nrows, "neume.clivis2")
    point = reference_point(img, g, avg_punctum=12.0)

    # avg_punctum 12 -> crop width = round(12 * 0.8) = 10, starting at ulx.
    assert point.x == ulx + 10 / 2
    assert point.x != g.center_x          # bbox center is 20px further right
    assert point.y == uly + reference_row(img, g, avg_punctum=12.0)


def test_reference_point_is_none_when_nothing_could_be_measured():
    """None, not a silent 0.0: a caller anchoring a whole neume needs to know
    the difference between "no measurement" and "measured at the top edge"."""
    img = blank_page()

    tiny = make_glyph(0, 5, 5, 8, 8, "neume.punctum")
    assert reference_point(img, tiny, avg_punctum=15.0) is None

    # Big enough to analyze, but blank parchment -- no ink to centroid.
    blank = make_glyph(1, 20, 20, 30, 30, "neume.punctum")
    assert reference_point(img, blank, avg_punctum=12.0) is None
    # reference_row, being Rodan-faithful, cannot say so and returns 0.0.
    assert reference_row(img, blank, avg_punctum=12.0) == 0.0

    off_image = make_glyph(2, 1000, 1000, 30, 30, "neume.punctum")
    assert reference_point(img, off_image, avg_punctum=12.0) is None

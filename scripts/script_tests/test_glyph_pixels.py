"""Sanity checks for pixel-level glyph analysis (glyph_pixels.py)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_io import Glyph
from glyph_pixels import (
    average_punctum, notehead_height, crop_and_binarize, row_projection_centroid,
    reference_row, reference_point, reference_region, TOP_HEAD_DEPTH_FRACTION,
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


def test_notehead_height_is_a_height_and_excludes_the_virga():
    """average_punctum is a *width*, and using it as a depth made the
    first-head band ~40% too deep on McGill_MS234-064 (35 px against a 25 px
    notehead). virga is excluded because its bbox includes the stem."""
    glyphs = [
        make_glyph(0, 0, 0, 40, 24, "neume.punctum"),
        make_glyph(1, 0, 0, 40, 26, "neume.inclinatum"),
        make_glyph(2, 0, 0, 40, 90, "neume.virga"),      # stem: not a notehead height
        make_glyph(3, 0, 0, 40, 70, "neume.clivis2"),    # multi-note: not counted
    ]
    assert notehead_height(glyphs) == 25.0
    assert notehead_height([]) == 0.0


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


def make_podatus_page(with_staff_line=False):
    """A podatus drawn the way the manuscripts draw it, with the two things
    that pulled its anchor below its first notehead:

        note2 (high, right)                rows  0-11
        |  ligature stroke on the right, descending PAST note 1
        note1 (low, left)                  rows 24-35
        |  (optional staff line crossing)  rows 42-44
        stroke foot                        rows      -47  <- sets the bbox bottom

    The bbox bottom is the stroke's, 12 rows below note 1, so a band hanging
    off the bbox bottom clips note 1's top off and centroids low. Returns
    (image, glyph, avg_punctum, notehead_h).
    """
    img = blank_page(120, 120)
    ulx, uly, ncols, nrows = 20, 20, 36, 48
    img[uly:uly + 12, ulx + 22:ulx + 32] = INK              # note 2, highest, right
    img[uly:uly + 48, ulx + 26:ulx + 29] = INK              # ligature stroke, right side
    img[uly + 24:uly + 36, ulx:ulx + 10] = INK              # note 1, bottom-left
    if with_staff_line:
        img[uly + 42:uly + 45, :] = INK                     # staff line, full width
    return img, make_glyph(0, ulx, uly, ncols, nrows, "neume.podatus2b"), 12.0, 12.0


def test_podatus_reference_is_its_first_notehead_not_the_bbox_bottom():
    """The reported bug: a podatus's anchor sat a little below its notehead.

    The band was positioned from the bbox's bottom edge, but that edge belongs
    to the descending ligature stroke, not to the first head -- so the band
    slid down past the head and kept only its lower rows. Measured at 0.27
    steps low on McGill_MS234-064's podatus2b (worst case 0.96), against the
    0.5-step threshold at which the pitch is simply wrong.
    """
    img, g, avg_punctum, nh = make_podatus_page()

    region = reference_region(img, g, avg_punctum, extended_rules=True, notehead_h=nh)
    offset = (region.uly - g.uly) + row_projection_centroid(
        crop_and_binarize(img, region.ulx, region.uly, region.ncols, region.nrows))

    print(f"podatus first-head offset {offset:.2f}; region rows "
          f"{region.uly - g.uly}..{region.uly - g.uly + region.nrows - 1}")

    assert region.region == REGION_BOTTOM_LEFT
    assert 24 <= offset <= 36           # inside note 1 (rows 24-35)

    # A band measured from the bbox bottom (rows 36-47) holds no note-1 ink at
    # all here; where it does overlap the head it keeps only its bottom rows.
    # That is the bias this replaced, and Rodan's path still has it.
    bbox_band = reference_row(img, g, avg_punctum)
    print(f"bbox-bottom band offset {bbox_band:.2f}")
    assert bbox_band > offset


def test_podatus_reference_ignores_a_staff_line_crossing_the_band():
    """Staff lines ink the column band just as densely as a notehead does.

    They are only 2-3 rows against a head's 12+, so weighing each run of ink
    tells them apart -- and it has to, because a staff line below the first
    head is exactly what a fixed-depth band averages into the anchor.
    """
    img, g, avg_punctum, nh = make_podatus_page(with_staff_line=True)

    region = reference_region(img, g, avg_punctum, extended_rules=True, notehead_h=nh)
    offset = (region.uly - g.uly) + row_projection_centroid(
        crop_and_binarize(img, region.ulx, region.uly, region.ncols, region.nrows))

    print(f"podatus offset with a staff line crossing: {offset:.2f}")
    assert 24 <= offset <= 36           # still inside note 1, not pulled to row 43

    clean_img, _, _, _ = make_podatus_page()
    clean = reference_region(clean_img, g, avg_punctum, extended_rules=True, notehead_h=nh)
    assert (region.uly, region.nrows) == (clean.uly, clean.nrows)


def test_first_head_band_is_capped_at_one_notehead_deep():
    """When a head is fused to its own stroke the band is one long run, so the
    depth cap is the only thing keeping the stroke out of the centroid."""
    img = blank_page(120, 120)
    ulx, uly, ncols, nrows = 20, 20, 36, 48
    img[uly:uly + 36, ulx:ulx + 10] = INK        # head and stroke, one solid run
    g = make_glyph(0, ulx, uly, ncols, nrows, "neume.podatus2b")

    shallow = reference_region(img, g, 12.0, extended_rules=True, notehead_h=12.0)
    deep = reference_region(img, g, 12.0, extended_rules=True, notehead_h=30.0)

    print(f"depth 12 -> {shallow.nrows} rows, depth 30 -> {deep.nrows} rows")
    assert shallow.nrows == 12
    assert deep.nrows == 30
    # Both end at the run's bottom (row 35); the cap only moves the top.
    assert shallow.uly + shallow.nrows == deep.uly + deep.nrows


def test_rodan_path_keeps_its_own_podatus_behavior():
    """reference_row is rodan_pitch_finder's, and a baseline that quietly
    adopts this module's rules stops being a baseline. Rodan crops podatus2b
    to a band on the bbox's bottom edge; that must not change."""
    img, g, avg_punctum, nh = make_podatus_page()

    rodan = reference_region(img, g, avg_punctum)          # extended_rules off
    assert rodan.region == REGION_BOTTOM_LEFT
    assert rodan.uly + rodan.nrows == g.uly + g.nrows      # flush with the bbox bottom

    # ...and passing notehead_h cannot reach it either.
    assert reference_region(img, g, avg_punctum, notehead_h=nh) == rodan


def make_clivis_page():
    """A clivis drawn as the manuscripts draw it -- a Pi, not two loose heads:

        note1 (top bar, spans the width)   rows  0-11
        |  left stem, FUSED to the bar and inside the left band
        |                          right stem
        |                          note2 (foot)   rows 48-59

    The left band's ink is therefore one run 60 rows long, five times a
    notehead, so the depth cap is the only thing separating bar from stem.
    Returns (image, glyph, avg_punctum, notehead_h).
    """
    img = blank_page(120, 120)
    ulx, uly, ncols, nrows = 20, 20, 36, 60
    img[uly:uly + 12, ulx:ulx + 32] = INK                   # note 1, the top bar
    img[uly + 12:uly + 48, ulx + 1:ulx + 5] = INK           # left stem, in the band
    img[uly + 12:uly + 48, ulx + 24:ulx + 28] = INK         # right stem
    img[uly + 48:uly + 60, ulx + 22:ulx + 32] = INK         # note 2, the foot
    return img, make_glyph(0, ulx, uly, ncols, nrows, "neume.clivis2"), 12.0, 12.0


def test_clivis_reference_is_its_top_notehead():
    """A clivis descends, so its first note is the top-left head.

    The midpoint anchor this replaced read the left band's whole-height
    centroid as the point *between* the two notes, which put the first note
    half the neume's span too high -- 1.15 steps for `clivis4b` on
    McGill_MS234-064, far enough to leave the glyph's own bbox.
    """
    img, g, avg_punctum, nh = make_clivis_page()

    region = reference_region(img, g, avg_punctum, extended_rules=True, notehead_h=nh)
    offset = (region.uly - g.uly) + row_projection_centroid(
        crop_and_binarize(img, region.ulx, region.uly, region.ncols, region.nrows))

    print(f"clivis top-head offset {offset:.2f}; region rows "
          f"{region.uly - g.uly}..{region.uly - g.uly + region.nrows - 1}")

    # REGION_TOP so _anchor_interval binds it to max(intervals) -- for a
    # descending neume that is the first note.
    assert region.region == REGION_TOP
    assert 0 <= offset <= 12            # inside the top bar (rows 0-11)

    # The full-height column average is what the midpoint rule measured, and it
    # sits down among the stems -- a third of the way down the glyph or worse.
    naive = reference_row(img, make_glyph(1, g.ulx, g.uly, g.ncols, g.nrows,
                                          "neume.oblique2"), avg_punctum)
    print(f"full-height column offset {naive:.2f}")
    assert offset < naive - 5


def test_clivis_crop_is_capped_tighter_than_a_whole_notehead():
    """The cap is load-bearing here, unlike on the ascending ligatures.

    A podatus keeps its stroke on the right, outside the left band, so the
    band's lowest run is the bare head and the cap rarely engages. A clivis's
    left stem is *inside* the band and fused to the head, so every pixel of the
    cap moves the anchor -- which is why it is a fraction of a notehead.
    """
    img, g, avg_punctum, nh = make_clivis_page()
    region = reference_region(img, g, avg_punctum, extended_rules=True, notehead_h=nh)

    assert region.nrows == max(1, round(nh * TOP_HEAD_DEPTH_FRACTION))
    assert region.uly == g.uly          # anchored to the run's top, not the bbox's
    assert TOP_HEAD_DEPTH_FRACTION < 1.0


def test_rodan_path_keeps_its_own_clivis_behavior():
    """clivis has no crop rule in Rodan at all -- it takes the default
    full-height left band. reference_row must still do exactly that."""
    img, g, avg_punctum, nh = make_clivis_page()

    rodan = reference_region(img, g, avg_punctum)          # extended_rules off
    assert rodan.region == REGION_FULL
    assert (rodan.uly, rodan.nrows) == (g.uly, g.nrows)   # the whole bbox height
    assert reference_region(img, g, avg_punctum, notehead_h=nh) == rodan


def make_torculus_page():
    """A torculus23-shaped glyph: up a second, then down a second past where it
    started, so the bbox's lowest ink is the THIRD note, not the first.

        note2 (high, middle)        rows  0-11
        |  connector down the left band
        note1 (left)                rows 30-41
                  note3 (right)     rows 48-59   <- sets the bbox bottom

    Returns (image, glyph, avg_punctum). Heads are drawn narrower than the crop
    so Otsu has background to threshold against.
    """
    img = blank_page(120, 120)
    ulx, uly, ncols, nrows = 20, 20, 36, 60
    img[uly:uly + 12, ulx + 12:ulx + 22] = INK             # note 2, highest
    img[uly:uly + 30, ulx + 6:ulx + 8] = INK               # connector, in the left band
    img[uly + 30:uly + 42, ulx:ulx + 8] = INK              # note 1, bottom-left
    img[uly + 48:uly + 60, ulx + 24:ulx + 32] = INK        # note 3, lowest
    return img, make_glyph(0, ulx, uly, ncols, nrows, "neume.torculus23"), 12.0


def test_torculus_reference_is_its_first_notehead():
    """The torculus rule has to isolate note 1 without either the ascending
    connector above it (which the old full-height column average included) or
    the bbox's bottom edge below it (which for this subclass is note 3, a step
    lower and on the far side of the glyph)."""
    img, g, avg_punctum = make_torculus_page()

    region = reference_region(img, g, avg_punctum, extended_rules=True)
    offset = (region.uly - g.uly) + row_projection_centroid(
        crop_and_binarize(img, region.ulx, region.uly, region.ncols, region.nrows))

    print(f"torculus first-head offset {offset:.2f}; region rows "
          f"{region.uly - g.uly}..{region.uly - g.uly + region.nrows - 1}")

    assert region.region == REGION_BOTTOM_LEFT
    assert 30 <= offset <= 43           # inside note 1 (rows 30-41)

    # Not dragged up by the connector: that is what the untrimmed full-height
    # left band does, and it is what this rule replaced.
    naive = reference_row(img, make_glyph(1, g.ulx, g.uly, g.ncols, g.nrows,
                                          "neume.clivis2"), avg_punctum)
    print(f"full-height column offset {naive:.2f}")
    assert offset > naive + 5

    # Nor pulled down to note 3: a band measured from the bbox bottom edge (the
    # podatus rule) lands on rows 48-59, two steps below note 1.
    assert offset < 46


def test_rodan_path_keeps_its_own_torculus_behavior():
    """reference_row feeds rodan_pitch_finder, which is only useful as a
    comparison if it keeps Rodan's own crop rules -- and Rodan has no torculus
    rule. The extra rule must not leak onto that path."""
    img, g, avg_punctum = make_torculus_page()

    assert reference_region(img, g, avg_punctum).region == REGION_FULL
    # Same value a class with no crop rule of its own gets: the default crop.
    plain = make_glyph(1, g.ulx, g.uly, g.ncols, g.nrows, "neume.clivis2")
    assert reference_row(img, g, avg_punctum) == reference_row(img, plain, avg_punctum)


def test_torculus_with_no_ink_in_the_left_band_is_unmeasurable():
    """An empty left band must not report a confident first-notehead region --
    reference_point's None path is what makes the caller fall back to geometry
    instead of anchoring three notes on nothing."""
    img = blank_page(120, 120)
    ulx, uly, ncols, nrows = 20, 20, 36, 40
    img[uly:uly + 12, ulx + 24:ulx + 34] = INK    # ink only on the right

    g = make_glyph(0, ulx, uly, ncols, nrows, "neume.torculus22")
    region = reference_region(img, g, avg_punctum=12.0, extended_rules=True)

    assert region.region == REGION_BOTTOM_LEFT
    assert region.nrows == nrows                   # untrimmed: nothing to trim to
    assert reference_point(img, g, avg_punctum=12.0) is None


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
        # oblique is the one multi-note class with no crop rule of its own.
        ("neume.oblique2", REGION_FULL),
        # Only under extended_rules; see the rodan-path tests.
        ("neume.torculus33", REGION_BOTTOM_LEFT),
        ("neume.clivis2", REGION_TOP),      # descends: first note is also highest
    ]
    for class_name, expected in cases:
        g = make_glyph(0, 20, 20, 30, 40, class_name)
        region = reference_region(img, g, avg_punctum=12.0, extended_rules=True)
        assert region is not None
        assert region.region == expected, f"{class_name} -> {region.region}"


def test_reference_point_x_is_the_crops_own_center():
    """The centroid y was measured from the crop's columns, so x has to be that
    band's center -- not the bbox center (which for a wide ligature is a column
    the measurement never looked at)."""
    img = blank_page()
    ulx, uly, ncols, nrows = 20, 20, 40, 30
    img[uly + 5:uly + 25, ulx + 1:ulx + 9] = INK

    # oblique2 takes the default full-height crop on both entry points, so the
    # two agree on y and this test is only about x.
    g = make_glyph(0, ulx, uly, ncols, nrows, "neume.oblique2")
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

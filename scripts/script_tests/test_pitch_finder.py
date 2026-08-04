"""End-to-end sanity checks for pitch_finder, against a synthetic 4-line stave.

Stave geometry: 4 flat lines 20px apart, x in [0,100], within_stave_index
0 (top, y=100) .. 3 (bottom, y=160). Per staff_io's step convention, the
bottom detected line is step 0 and each line up is +2 steps, so:
  y=160 -> step 0 (bottom line)
  y=140 -> step 2
  y=120 -> step 4
  y=100 -> step 6 (top line)
and step decreases linearly by 1 per 10px going down.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_io import Glyph
from staff_io import StaffLine, Stave
from neume_shapes import NeumeShapeTable
from pitch_finder import find_pitches

BACKGROUND = 220  # light parchment
INK = 20          # dark ink


def make_stave():
    lines = []
    for within_idx, y in enumerate([100, 120, 140, 160]):
        lines.append(StaffLine(
            line_id=f"line{within_idx}", stave_id=0, within_stave_index=within_idx,
            x_start=0, x_end=100, y_values=[float(y)] * 101, scale_unit=10.0,
        ))
    return Stave(stave_id=0, lines=lines)


def make_shapes():
    return NeumeShapeTable(
        neume_intervals={
            "neume.punctum": [0],
            "neume.clivis2": [0, -1],
            "neume.virga": [0],
            "neume.podatus3": [0, 2],
            "neume.torculus22": [0, 1, 0],
        },
        clef_classes={"clef.c": "C"},
        pitchless_classes=set(),
    )


def blank_page(width=200, height=200):
    return np.full((height, width), BACKGROUND, dtype=np.uint8)


def make_clef():
    """A clef.c on the y=140 line (step 2). Small enough (10x8) that pixel
    analysis is skipped for it, so it anchors geometrically in both modes and
    the same pitch reference applies to every test below."""
    return Glyph(index=0, ulx=0, uly=136, nrows=8, ncols=10, class_name="clef.c",
                 confidence=0.9, state="AUTOMATIC")


def test_clef_and_note_on_same_stave_resolve_pitch():
    staves = [make_stave()]
    shapes = make_shapes()

    clef = Glyph(index=0, ulx=0, uly=136, nrows=8, ncols=10, class_name="clef.c",
                 confidence=0.9, state="AUTOMATIC")
    # Punctum sits on the top line (step 6), to the right of the clef.
    note = Glyph(index=1, ulx=50, uly=96, nrows=8, ncols=10, class_name="neume.punctum",
                 confidence=0.9, state="AUTOMATIC")

    results = find_pitches([clef, note], staves, shapes)
    clef_result, note_result = results

    print(f"clef step: {clef_result.note_components[0].stave_step}")
    assert clef_result.reason is None
    assert clef_result.note_components[0].pitch == {"pname": "C", "oct": 4}

    print(f"note step: {note_result.note_components[0].stave_step}, pitch: {note_result.note_components[0].pitch}")
    assert note_result.reason is None
    assert note_result.stave_id == 0
    assert round(note_result.note_components[0].stave_step) == 6
    assert note_result.note_components[0].pitch == {"pname": "G", "oct": 4}


def test_multi_note_neume_decomposes_into_separate_pitches():
    staves = [make_stave()]
    shapes = make_shapes()

    clef = Glyph(index=0, ulx=0, uly=136, nrows=8, ncols=10, class_name="clef.c",
                 confidence=0.9, state="AUTOMATIC")
    # clivis2 = [0, -1]; place bbox so top edge = step 5, bottom edge = step 4.
    clivis = Glyph(index=1, ulx=50, uly=110, nrows=10, ncols=10, class_name="neume.clivis2",
                   confidence=0.9, state="AUTOMATIC")

    results = find_pitches([clef, clivis], staves, shapes)
    clivis_result = results[1]

    steps = [round(nc.stave_step) for nc in clivis_result.note_components]
    pitches = [nc.pitch for nc in clivis_result.note_components]
    print(f"clivis2 steps: {steps}, pitches: {pitches}")

    assert steps == [5, 4]
    assert pitches == [{"pname": "F", "oct": 4}, {"pname": "E", "oct": 4}]


def test_each_note_component_records_its_own_pixel_center():
    """Every component carries the page-pixel point its pitch was read from,
    so the debug overlay can mark each notehead center of a multi-note neume
    (and so a wrong pitch can be traced to a mis-placed center)."""
    staves = [make_stave()]
    shapes = make_shapes()
    stave = staves[0]

    clef = Glyph(index=0, ulx=0, uly=136, nrows=8, ncols=10, class_name="clef.c",
                 confidence=0.9, state="AUTOMATIC")
    clivis = Glyph(index=1, ulx=50, uly=110, nrows=10, ncols=10, class_name="neume.clivis2",
                   confidence=0.9, state="AUTOMATIC")

    components = find_pitches([clef, clivis], staves, shapes)[1].note_components
    centers = [(nc.center_x, nc.center_y) for nc in components]
    print(f"clivis2 centers: {centers}")

    # x is the glyph's horizontal center (the x the stave was queried at)...
    assert all(nc.center_x == clivis.center_x for nc in components)
    # ...and y is each component's own step converted back to pixels. steps
    # [5, 4] on this stave (1 step = 10px, step 6 = y100) are y=110 and y=120,
    # i.e. this clivis's bbox top and bottom edges.
    assert [nc.center_y for nc in components] == [110.0, 120.0]
    for nc in components:
        assert nc.center_y == stave.y_at_step(nc.center_x, nc.stave_step)


def test_missing_clef_still_reports_stave_and_step():
    staves = [make_stave()]
    shapes = make_shapes()

    note = Glyph(index=0, ulx=50, uly=96, nrows=8, ncols=10, class_name="neume.punctum",
                 confidence=0.9, state="AUTOMATIC")

    results = find_pitches([note], staves, shapes)
    result = results[0]

    assert result.reason == "missing_clef"
    assert result.stave_id == 0
    assert result.note_components[0].pitch is None
    assert round(result.note_components[0].stave_step) == 6


def test_pitchless_text_glyph_short_circuits():
    staves = [make_stave()]
    shapes = make_shapes()

    text_glyph = Glyph(index=0, ulx=50, uly=1000, nrows=20, ncols=50, class_name="text",
                        confidence=0.0, state="UNCLASSIFIED")

    results = find_pitches([text_glyph], staves, shapes)
    result = results[0]

    assert result.reason == "pitchless_symbol"
    assert result.stave_id is None
    assert result.note_components == []


def test_unknown_class_falls_back_to_single_note_approximation():
    staves = [make_stave()]
    shapes = make_shapes()

    clef = Glyph(index=0, ulx=0, uly=136, nrows=8, ncols=10, class_name="clef.c",
                 confidence=0.9, state="AUTOMATIC")
    # "neume.clivis1" is not in make_shapes()'s neume_intervals -- not a
    # clef, not pitchless, just missing from the CSV-derived table.
    mystery = Glyph(index=1, ulx=50, uly=96, nrows=8, ncols=10, class_name="neume.clivis1",
                     confidence=0.9, state="AUTOMATIC")

    results = find_pitches([clef, mystery], staves, shapes)
    mystery_result = results[1]

    assert mystery_result.reason is None
    assert mystery_result.note_components[0].pitch is not None
    assert "approximate_unknown_shape" in mystery_result.flags
    # Single-note fallback == bbox top/bottom center, same as punctum: step 6.
    assert round(mystery_result.note_components[0].stave_step) == 6


def test_pixel_anchor_reads_a_virga_from_its_notehead_not_its_stem():
    """The whole point of borrowing rodan's notehead finding: a virga's stem
    makes the bbox span a lie, and the bbox-span anchor lands nearly 3 steps
    below the head that actually carries the pitch."""
    staves, shapes = [make_stave()], make_shapes()

    # Notehead centered on the top line (y=100, step 6), stem trailing to y=159.
    # Drawn narrower than the crop so Otsu has background to threshold against.
    img = blank_page()
    img[94:107, 51:59] = INK      # notehead
    img[107:160, 55:57] = INK     # stem

    clef = make_clef()
    virga = Glyph(index=1, ulx=50, uly=94, nrows=66, ncols=12,
                  class_name="neume.virga", confidence=0.9, state="AUTOMATIC")

    pixel = find_pitches([clef, virga], staves, shapes, image=img)[1]
    bbox = find_pitches([clef, virga], staves, shapes)[1]
    pixel_step = pixel.note_components[0].stave_step
    bbox_step = bbox.note_components[0].stave_step
    print(f"virga step: pixel {pixel_step:.2f} vs bbox-span {bbox_step:.2f}")

    # The head sits on the top line, so the anchor must round to step 6...
    assert round(pixel_step) == 6
    assert pixel.anchor.source == "pixel_centroid"
    assert pixel.anchor.region == "top"      # virga crop = top band
    assert pixel.anchor.interval == 0.0
    # ...while the bbox span, averaging head and stem, lands way low.
    assert bbox_step < 4
    assert bbox.anchor.source == "bbox_span"
    # A whole-pitch difference, not a rounding one: G4 (top line) vs D4.
    assert pixel.note_components[0].pitch == {"pname": "G", "oct": 4}
    assert bbox.note_components[0].pitch == {"pname": "D", "oct": 4}


def test_bottom_left_crop_anchors_the_podatus_lowest_note():
    """A bottom-left crop isolates the ligature's lower head, so it anchors
    min(intervals) and the interval table places the upper head -- landing it
    on the very ink the crop deliberately excluded."""
    staves, shapes = [make_stave()], make_shapes()

    # podatus3 = [0, +2]: lower head on y=140 (step 2), upper head on y=120
    # (step 4) and further right. The bottom-left crop sees only the former.
    img = blank_page()
    img[115:126, 62:72] = INK     # upper-right head
    img[135:146, 51:59] = INK     # lower-left head
    img[20:30, 11:19] = INK       # the punctum below, so avg_punctum is real

    clef = make_clef()
    punctum = Glyph(index=1, ulx=10, uly=20, nrows=10, ncols=12,
                    class_name="neume.punctum", confidence=0.9, state="AUTOMATIC")
    podatus = Glyph(index=2, ulx=50, uly=114, nrows=33, ncols=24,
                    class_name="neume.podatus3", confidence=0.9, state="AUTOMATIC")

    result = find_pitches([clef, punctum, podatus], staves, shapes, image=img)[2]
    steps = [round(nc.stave_step) for nc in result.note_components]
    print(f"podatus3 steps: {steps}, anchor: {result.anchor}")

    assert result.anchor.region == "bottom_left"
    assert result.anchor.interval == 0.0     # the crop IS the neume's first/lowest note
    assert steps == [2, 4]

    # bbox-span mode reads both edges instead and misses both heads by a step.
    bbox_steps = [round(nc.stave_step)
                  for nc in find_pitches([clef, punctum, podatus], staves, shapes)[2].note_components]
    print(f"podatus3 bbox-span steps: {bbox_steps}")
    assert bbox_steps == [1, 3]


def test_torculus_anchors_its_first_head_and_returns_to_that_pitch():
    """A torculus22 goes up a second and back down, so notes 1 and 3 share a
    pitch. Two things have to hold at once for that to come out right: the
    interval list must be cumulative ([0, 1, 0], not [0, 1, -1]), and the
    anchor must bind to note 1 rather than to the span's midpoint -- for
    [0, 1, 0] that midpoint is +0.5, which would drag all three notes half a
    step down and land them between staff positions.
    """
    staves, shapes = [make_stave()], make_shapes()

    # notes 1 and 3 on y=140 (step 2), note 2 on y=130 (step 3), plus the
    # ascending connector that used to drag the anchor upward.
    img = blank_page()
    img[135:146, 51:60] = INK     # note 1, bottom-left
    img[125:136, 57:60] = INK     # connector up to note 2
    img[125:136, 62:71] = INK     # note 2, highest
    img[135:146, 74:83] = INK     # note 3, back down to note 1's step
    img[20:30, 11:19] = INK       # punctum, for avg_punctum

    clef = make_clef()
    punctum = Glyph(index=1, ulx=10, uly=20, nrows=10, ncols=12,
                    class_name="neume.punctum", confidence=0.9, state="AUTOMATIC")
    torculus = Glyph(index=2, ulx=50, uly=125, nrows=21, ncols=34,
                     class_name="neume.torculus22", confidence=0.9, state="AUTOMATIC")

    result = find_pitches([clef, punctum, torculus], staves, shapes, image=img)[2]
    steps = [nc.stave_step for nc in result.note_components]
    pitches = [nc.pitch for nc in result.note_components]
    print(f"torculus22 steps: {[round(s, 2) for s in steps]}, pitches: {pitches}, "
          f"anchor: {result.anchor}")

    assert result.anchor.region == "bottom_left"
    assert result.anchor.interval == 0.0        # the crop IS note 1
    assert [round(s) for s in steps] == [2, 3, 2]
    assert steps[0] == steps[2]                 # the defining property
    assert pitches == [{"pname": "C", "oct": 4}, {"pname": "D", "oct": 4},
                       {"pname": "C", "oct": 4}]


def test_full_bbox_crop_anchors_the_middle_of_the_note_span():
    """For classes rodan has no crop rule for, the centroid covers the whole
    shape and belongs to no single head -- so it anchors the span's midpoint
    (a fractional interval), and the notes straddle it symmetrically."""
    staves, shapes = [make_stave()], make_shapes()

    # clivis2 = [0, -1]: heads on y=120 (step 4) and y=130 (step 3), drawn as
    # one block so the ink centroid sits midway between them.
    img = blank_page()
    img[115:136, 51:59] = INK
    img[20:30, 11:19] = INK       # punctum, for avg_punctum

    clef = make_clef()
    punctum = Glyph(index=1, ulx=10, uly=20, nrows=10, ncols=12,
                    class_name="neume.punctum", confidence=0.9, state="AUTOMATIC")
    clivis = Glyph(index=2, ulx=50, uly=114, nrows=36, ncols=14,
                   class_name="neume.clivis2", confidence=0.9, state="AUTOMATIC")

    result = find_pitches([clef, punctum, clivis], staves, shapes, image=img)[2]
    steps = [nc.stave_step for nc in result.note_components]
    print(f"clivis2 steps: {steps}, anchor step {result.anchor.stave_step:.2f}")

    assert result.anchor.region == "full"
    assert result.anchor.interval == -0.5             # midpoint of [0, -1]
    assert steps[0] - steps[1] == 1                   # exactly the CSV interval
    assert sum(steps) / 2 == result.anchor.stave_step  # straddling the centroid
    assert [round(s) for s in steps] == [4, 3]


def test_unmeasurable_ink_falls_back_to_geometry_and_says_so():
    """A crop with no ink in it must not anchor the neume at the bbox's top
    edge (what rodan's 0.0 centroid would mean here) -- fall back to the
    geometric anchor and flag it, so a page of silent fallbacks is visible."""
    staves, shapes = [make_stave()], make_shapes()
    blank = blank_page()   # glyphs sit on empty parchment

    clef = make_clef()
    virga = Glyph(index=1, ulx=50, uly=94, nrows=66, ncols=12,
                  class_name="neume.virga", confidence=0.9, state="AUTOMATIC")

    with_image = find_pitches([clef, virga], staves, shapes, image=blank)[1]
    without = find_pitches([clef, virga], staves, shapes)[1]

    assert "pixel_anchor_unavailable" in with_image.flags
    assert with_image.anchor.source == "bbox_span"
    assert with_image.note_components[0].stave_step == without.note_components[0].stave_step


def test_page_with_no_punctum_or_virga_cannot_size_the_crops():
    """avg_punctum is the crop scale; with no punctum/virga on the page it is 0
    and the crops would collapse to a 1px sliver. Fall back to geometry."""
    staves, shapes = [make_stave()], make_shapes()
    img = blank_page()
    img[115:136, 51:59] = INK

    clef = make_clef()
    clivis = Glyph(index=1, ulx=50, uly=114, nrows=36, ncols=14,
                   class_name="neume.clivis2", confidence=0.9, state="AUTOMATIC")

    result = find_pitches([clef, clivis], staves, shapes, image=img)[1]
    assert "pixel_anchor_unavailable" in result.flags
    assert result.anchor.source == "bbox_span"


def test_glyph_far_from_any_stave_is_missing_staff():
    staves = [make_stave()]
    shapes = make_shapes()

    far_glyph = Glyph(index=0, ulx=50, uly=5000, nrows=8, ncols=10, class_name="neume.punctum",
                       confidence=0.9, state="AUTOMATIC")

    results = find_pitches([far_glyph], staves, shapes)
    result = results[0]

    assert result.reason == "missing_staff"
    assert result.stave_id is None

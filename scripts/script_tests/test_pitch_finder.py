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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_io import Glyph
from staff_io import StaffLine, Stave
from neume_shapes import NeumeShapeTable
from pitch_finder import find_pitches


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
        },
        clef_classes={"clef.c": "C"},
        pitchless_classes=set(),
    )


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


def test_glyph_far_from_any_stave_is_missing_staff():
    staves = [make_stave()]
    shapes = make_shapes()

    far_glyph = Glyph(index=0, ulx=50, uly=5000, nrows=8, ncols=10, class_name="neume.punctum",
                       confidence=0.9, state="AUTOMATIC")

    results = find_pitches([far_glyph], staves, shapes)
    result = results[0]

    assert result.reason == "missing_staff"
    assert result.stave_id is None

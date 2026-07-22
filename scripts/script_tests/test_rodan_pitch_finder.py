"""Sanity checks for rodan_pitch_finder.py against synthetic staves/glyphs.

Stave geometry mirrors test_staff_io.py / test_pitch_finder.py: 4 flat
lines 20px apart, within_stave_index 0 (top, y=100) .. 3 (bottom, y=160).
Per staff_io's step convention: y=160 -> step 0 (bottom line), y=100 ->
step 6 (top line), 1 step = 10px.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_io import Glyph
from staff_io import StaffLine, Stave
import clef_rules
import rodan_pitch_finder as rpf
from rodan_pitch_finder import assign_stave, _stave_position, find_pitches_rodan

BACKGROUND = 220
INK = 20


def make_stave(stave_id=0, x_start=0, x_end=100, ys=(100, 120, 140, 160)):
    lines = [
        StaffLine(line_id=f"s{stave_id}_l{i}", stave_id=stave_id, within_stave_index=i,
                  x_start=x_start, x_end=x_end, y_values=[float(y)] * (x_end - x_start + 1),
                  scale_unit=10.0)
        for i, y in enumerate(ys)
    ]
    return Stave(stave_id=stave_id, lines=lines)


def make_glyph(index, ulx, uly, ncols, nrows, class_name, state="AUTOMATIC", confidence=0.9):
    return Glyph(index=index, ulx=ulx, uly=uly, nrows=nrows, ncols=ncols,
                 class_name=class_name, confidence=confidence, state=state)


def blank_page(width=400, height=400):
    return np.full((height, width), BACKGROUND, dtype=np.uint8)


# ---------------------------------------------------------------------
# _stave_position: rounding == space_proportion=0.5, clef snaps to a line,
# and extrapolation continues past the real lines instead of clamping.
# ---------------------------------------------------------------------

def test_stave_position_on_a_line():
    stave = make_stave()
    step, flags = _stave_position(50, 140.0, stave, is_clef=False)
    assert step == 2
    assert flags == []


def test_stave_position_in_a_space():
    stave = make_stave()
    step, _ = _stave_position(50, 130.0, stave, is_clef=False)  # midway between step2/step4
    assert step == 3


def test_stave_position_near_line_rounds_to_that_line():
    stave = make_stave()
    # y=132 -> continuous step = 2.8 -> rounds to 3 (space); still within
    # the "space_proportion" middle band, not snapped to the line.
    step, _ = _stave_position(50, 132.0, stave, is_clef=False)
    assert step == 3


def test_clef_snaps_to_nearest_line_not_a_space():
    stave = make_stave()
    # y=130 is exactly the space (step 3); a clef there must snap to the
    # nearest LINE (step 2 or 4), never land on step 3.
    step, _ = _stave_position(50, 130.0, stave, is_clef=True)
    assert step in (2, 4)
    assert step % 2 == 0


def test_stave_position_extrapolates_above_and_below():
    stave = make_stave()
    above, _ = _stave_position(50, 80.0, stave, is_clef=False)   # 20px above top line (step6) -> step8
    below, _ = _stave_position(50, 180.0, stave, is_clef=False)  # 20px below bottom line (step0) -> step-2
    assert above == 8
    assert below == -2


# ---------------------------------------------------------------------
# assign_stave: 3-tier fallback
# ---------------------------------------------------------------------

def test_assign_stave_bbox_intersection_wins():
    stave = make_stave()
    g = make_glyph(0, 40, 90, 20, 80, "neume.punctum")  # bbox clearly overlaps stave bbox
    stave_found, flags = assign_stave(g, [stave], avg_punctum=15.0)
    assert stave_found is stave
    assert flags == []


def test_assign_stave_y_bound_fallback():
    # Glyph's x-range doesn't overlap the stave's x-range at all (so no bbox
    # intersection), but it's within the y-margin -- should fall back to
    # tier 2 rather than report missing_staff.
    stave = make_stave(x_start=0, x_end=100)
    g = make_glyph(0, 500, 120, 10, 10, "neume.punctum")
    stave_found, flags = assign_stave(g, [stave], avg_punctum=15.0)
    assert stave_found is stave
    assert flags == ["y_bound_fallback"]


def test_assign_stave_missing_when_far_from_everything():
    stave = make_stave()
    g = make_glyph(0, 500, 5000, 10, 10, "neume.punctum")
    stave_found, flags = assign_stave(g, [stave], avg_punctum=15.0)
    assert stave_found is None
    assert flags == ["missing_staff"]


def test_assign_stave_forced_nearest_when_enabled():
    stave = make_stave()
    g = make_glyph(0, 500, 5000, 10, 10, "neume.punctum")
    rpf.ALWAYS_FIND_STAFF_NO = True
    try:
        stave_found, flags = assign_stave(g, [stave], avg_punctum=15.0)
    finally:
        rpf.ALWAYS_FIND_STAFF_NO = False
    assert stave_found is stave
    assert flags == ["forced_nearest"]


# ---------------------------------------------------------------------
# clef_rules.step_to_pitch cross-check against a real Rodan record
# (from 238r-heuristic_pitch_finding.json, verified by hand): a glyph with
# clef=clef.c at clef_line=7 and strt_pos=10 resolves to note=g, octave=3.
# Rodan's own position numbering increases DOWN the page (opposite sign
# from our step convention, which increases with pitch), so the equivalent
# delta in our convention is (clef_line - strt_pos).
# ---------------------------------------------------------------------

def test_step_to_pitch_matches_real_rodan_record():
    assert clef_rules.step_to_pitch(7 - 10, "C", 4) == ("G", 3)


# ---------------------------------------------------------------------
# find_pitches_rodan: end-to-end, including the page-wide clef propagation
# that's deliberately different from pitch_finder.py's per-stave isolation.
# ---------------------------------------------------------------------

def test_clef_propagates_to_a_later_stave_with_no_clef_of_its_own():
    stave0 = make_stave(stave_id=0, x_start=0, x_end=100, ys=(100, 120, 140, 160))
    stave1 = make_stave(stave_id=1, x_start=0, x_end=100, ys=(300, 320, 340, 360))
    img = blank_page()

    clef = make_glyph(0, 10, 136, 10, 8, "clef.c")           # stave 0, lands ~step2 (line140)
    note0 = make_glyph(1, 50, 96, 10, 8, "neume.punctum")    # stave 0, top line (step6)
    note1 = make_glyph(2, 50, 296, 10, 8, "neume.punctum")   # stave 1, no clef of its own

    results = find_pitches_rodan([clef, note0, note1], [stave0, stave1], img)
    by_index = {r.glyph_index: r for r in results}

    assert by_index[0].pitch is not None          # clef itself gets a pitch label
    assert by_index[1].reason is None
    assert by_index[2].reason is None              # carried the clef from stave 0
    assert by_index[2].pitch is not None


def test_note_before_any_clef_is_missing_clef():
    stave = make_stave()
    img = blank_page()
    note = make_glyph(0, 50, 96, 10, 8, "neume.punctum")

    results = find_pitches_rodan([note], [stave], img)
    assert results[0].reason == "missing_clef"
    assert results[0].pitch is None
    assert results[0].stave_step is not None  # position is still reported for debugging


def test_pitchless_and_unclassified_short_circuit():
    stave = make_stave()
    img = blank_page()
    divisio = make_glyph(0, 50, 130, 10, 8, "divisio.maxima")
    text_glyph = make_glyph(1, 50, 130, 10, 8, "text", state="UNCLASSIFIED", confidence=0.0)
    skip_glyph = make_glyph(2, 50, 130, 10, 8, "skip.dot")

    results = find_pitches_rodan([divisio, text_glyph, skip_glyph], [stave], img)
    assert all(r.reason == "pitchless_symbol" for r in results)


def test_virga_pixel_reference_feeds_through_to_a_real_pitch():
    stave = make_stave()
    img = blank_page()
    # Notehead near the top line (step6 at y~100), long stem trailing down
    # so a naive bbox-center reference would land much lower.
    clef = make_glyph(0, 10, 136, 10, 8, "clef.c")
    virga_ulx, virga_uly, virga_ncols, virga_nrows = 50, 94, 10, 40
    img[virga_uly:virga_uly + 10, virga_ulx:virga_ulx + virga_ncols] = INK   # notehead ~ y 94-103
    img[virga_uly + 10:virga_uly + virga_nrows, virga_ulx + 4:virga_ulx + 6] = INK  # stem below
    virga = make_glyph(1, virga_ulx, virga_uly, virga_ncols, virga_nrows, "neume.virga")

    results = find_pitches_rodan([clef, virga], [stave], img)
    virga_result = [r for r in results if r.glyph_index == 1][0]

    assert virga_result.reason is None
    assert virga_result.pitch is not None
    # Reference should land near the notehead (top line, step ~6), not
    # dragged down toward the stem's midpoint (which would read much lower).
    assert virga_result.stave_step >= 4

"""Checks that stave grouping is re-derived correctly from line geometry.

The case that matters is the two-column page: staff-finding groups by y alone,
so a left-column stave and a right-column stave at the same height arrive as one
stave of eight lines, which staff_io then reads as a 14-step staff. Each test
builds the geometry by hand so the expected grouping is unambiguous.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from staff_io import Stave, StaffLine
from staff_regroup import estimate_line_gap, regroup_entries, split_columns


def line(line_id: str, y: float, x_start: int, x_end: int, *, slope: float = 0.0,
         stave_id=None, within_stave_index=None, residual: float = 1.0) -> dict:
    """One staff-finding entry: a straight line of the given slope."""
    n = x_end - x_start + 1
    return {
        "id": line_id,
        "source": "detected",
        "centerline_page": {
            "x_start": x_start, "x_end": x_end,
            "y_values": [y + slope * i for i in range(n)],
        },
        "fit": {"residual_mean": residual},
        "quality": {"flags": []},
        "scale_unit": 16.5,
        "column_id": None,
        "stave_id": stave_id,
        "within_stave_index": within_stave_index,
    }


def four_line_stave(prefix: str, top_y: float, x_start: int, x_end: int,
                    gap: float = 14.0, **kwargs) -> list[dict]:
    return [line(f"{prefix}{i}", top_y + gap * i, x_start, x_end, **kwargs)
            for i in range(4)]


def grouping(entries: list[dict]) -> dict[int, list[int]]:
    """{stave_id: [within_stave_index, ...]} for the regrouped entries."""
    out: dict[int, list[int]] = {}
    for entry in entries:
        out.setdefault(entry["stave_id"], []).append(entry["within_stave_index"])
    return {sid: sorted(idxs) for sid, idxs in out.items()}


def test_two_column_page_splits_into_one_stave_per_column():
    # The bug: two 4-line staves at the same y, one per column. Staff-finding
    # hands these over as a single 8-line stave.
    entries = (four_line_stave("L", 100.0, 100, 400)
               + four_line_stave("R", 100.0, 500, 800))
    out, report = regroup_entries(entries)

    assert report.columns == 2
    assert report.staves == 2
    assert grouping(out) == {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
    # Column 0 is the left one, and no line crossed over.
    left = {e["id"] for e in out if e["stave_id"] == 0}
    assert left == {"L0", "L1", "L2", "L3"}


def test_two_column_stave_spans_six_steps_not_fourteen():
    # What the split is for: the highest step on a 4-line stave is 6.
    entries = (four_line_stave("L", 100.0, 100, 400)
               + four_line_stave("R", 100.0, 500, 800))
    out, _ = regroup_entries(entries)

    for stave_id, x in ((0, 250), (1, 650)):
        lines = [StaffLine(line_id=e["id"], stave_id=e["stave_id"],
                           within_stave_index=e["within_stave_index"],
                           x_start=e["centerline_page"]["x_start"],
                           x_end=e["centerline_page"]["x_end"],
                           y_values=e["centerline_page"]["y_values"],
                           scale_unit=e["scale_unit"])
                 for e in out if e["stave_id"] == stave_id]
        steps = sorted(step for step, _ in Stave(stave_id, lines).step_at_x(x))
        assert steps == [0, 2, 4, 6]


def test_full_width_line_does_not_hide_the_gutter():
    # A page-top rule spanning both columns is why the gutter is found from
    # coverage counts rather than from x that nothing covers at all.
    entries = ([line("rule", 20.0, 100, 800)]
               + four_line_stave("L", 100.0, 100, 400)
               + four_line_stave("R", 100.0, 500, 800))
    _, report = regroup_entries(entries)
    assert report.columns == 2


def test_single_column_page_is_left_alone():
    entries = (four_line_stave("A", 100.0, 100, 800)
               + four_line_stave("B", 300.0, 100, 800))
    out, report = regroup_entries(entries)

    assert report.columns == 1
    assert grouping(out) == {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}
    assert report.lines_kept == 8
    assert report.incomplete == []


def test_fragments_split_by_an_initial_share_one_line_index():
    # A decorated initial in the middle of a stave splits every line in two.
    # The halves are a few px apart vertically -- far less than a line gap --
    # so they are the same physical line, not eight lines. The second, unbroken
    # stave is what keeps the initial's gap from reading as a column gutter;
    # see test_initial_wider_than_a_gutter_needs_a_stave_to_bridge_it.
    entries = (four_line_stave("left", 100.0, 470, 660)
               + four_line_stave("right", 103.5, 710, 800)
               + four_line_stave("below", 300.0, 470, 800))
    out, report = regroup_entries(entries)

    assert report.columns == 1
    assert grouping(out) == {0: [0, 0, 1, 1, 2, 2, 3, 3], 1: [0, 1, 2, 3]}
    assert report.lines_kept == 12  # fragments are kept, not deduplicated away


def test_initial_wider_than_a_gutter_needs_a_stave_to_bridge_it():
    """Documents the one assumption column detection rests on.

    A gutter is found as x that few lines cover, which is what tells a real
    column gutter (no stave on the page crosses it) apart from the gap an
    initial leaves inside one stave (other staves cross it). With nothing else
    on the page to bridge it, a gap wider than MIN_GUTTER_PX is indistinguishable
    from a gutter and the stave is split. Every page here has staves that bridge,
    so this only bites a crop holding a single interrupted stave.
    """
    entries = (four_line_stave("left", 100.0, 470, 660)
               + four_line_stave("right", 103.5, 710, 800))
    _, report = regroup_entries(entries)
    assert report.columns == 2


def test_duplicate_detections_of_one_line_are_dropped():
    entries = four_line_stave("A", 100.0, 100, 800)
    # Same line found twice, the second slightly shorter and worse-fitting.
    entries.append(line("dup", 100.3, 120, 700, residual=3.0))
    out, report = regroup_entries(entries)

    assert report.lines_kept == 4
    assert grouping(out) == {0: [0, 1, 2, 3]}
    assert "dup" not in {e["id"] for e in out}


def test_undetected_line_leaves_a_hole_in_the_indices():
    # 3 of 4 lines detected, the missing one in the middle. Numbering the
    # survivors 0,1,2 would move every note on the stave by a step, so the
    # index comes from the spacing and index 1 goes unused.
    entries = [line("A0", 100.0, 100, 800),
               line("A2", 128.0, 100, 800),
               line("A3", 142.0, 100, 800)]
    out, report = regroup_entries(entries)

    assert grouping(out) == {0: [0, 2, 3]}
    assert report.incomplete == [(0, [0, 2, 3])]
    assert "missing a detected line" in report.summary()


def test_spurious_line_does_not_shift_the_real_ones():
    # A stray detection a third of a gap below the top line. The real lines
    # must keep indices 0..3 rather than all sliding down one.
    entries = four_line_stave("A", 100.0, 100, 800)
    entries.append(line("stray", 104.6, 810, 900))
    out, _ = regroup_entries(entries)

    by_id = {e["id"]: e["within_stave_index"] for e in out}
    assert [by_id[f"A{i}"] for i in range(4)] == [0, 1, 2, 3]


def test_sloped_lines_still_group_per_column():
    entries = (four_line_stave("L", 100.0, 100, 400, slope=-0.02)
               + four_line_stave("R", 100.0, 500, 800, slope=-0.02))
    _, report = regroup_entries(entries)
    assert (report.columns, report.staves) == (2, 2)
    assert report.line_counts == {4: 2}


def test_estimate_line_gap_ignores_fragment_offsets():
    # Every line detected as two offset fragments: a plain sort-and-difference
    # would measure the 3.5px offset, not the 14px gap.
    entries = (four_line_stave("left", 100.0, 470, 660)
               + four_line_stave("right", 103.5, 710, 800))
    assert estimate_line_gap(entries) == 14.0


def test_estimate_line_gap_ignores_duplicates():
    # A duplicate 0.4px off its original pulls the measurement to 13.6 rather
    # than 14.0 -- it is the nearest-neighbour distances that shift, not the
    # gap. Only the ~25% accuracy the index rounding needs is being claimed.
    entries = four_line_stave("A", 100.0, 100, 800)
    entries += [line(f"dup{i}", 100.0 + 14.0 * i + 0.4, 100, 800) for i in range(4)]
    assert estimate_line_gap(entries) == pytest.approx(14.0, abs=1.0)


def test_split_columns_keeps_a_narrow_gap_as_one_column():
    # Two staves with a 10px hole between them are one column, not two: a real
    # gutter is wide (MIN_GUTTER_PX), a word space is not.
    entries = four_line_stave("A", 100.0, 100, 400) + four_line_stave("B", 100.0, 410, 700)
    assert len(split_columns(entries)) == 1


def test_empty_input():
    out, report = regroup_entries([])
    assert out == []
    assert (report.staves, report.columns, report.lines_kept) == (0, 0, 0)

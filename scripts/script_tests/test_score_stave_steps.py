"""The two accuracy metrics and the hand-grouping detection behind score_stave_steps."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from score_stave_steps import is_manually_grouped, score_glyph


def test_exact_match_scores_every_note_and_the_glyph():
    assert score_glyph([3, 4, 3], [3, 4, 3]) == (3, True)


def test_one_note_off_loses_that_note_and_the_glyph():
    assert score_glyph([3, 5, 3], [3, 4, 3]) == (2, False)


def test_extra_predicted_note_keeps_matched_notes_but_fails_the_glyph():
    # e.g. IC calls a single punctum a clivis: the first note can still be right
    assert score_glyph([4, 3], [4]) == (1, False)


def test_missing_predicted_note_loses_its_point():
    assert score_glyph([4], [4, 3]) == (1, False)


def test_no_prediction_scores_nothing():
    assert score_glyph([], [2]) == (0, False)


def test_notes_are_matched_by_position_not_as_a_set():
    assert score_glyph([3, 2], [2, 3]) == (0, False)


def _write_staff_json(path: Path, flags_per_entry):
    entries = [{"id": f"line{i}", "quality": {"flags": flags}}
               for i, flags in enumerate(flags_per_entry)]
    path.write_text(json.dumps(entries))
    return path


def test_detects_fix_stafflines_grouping(tmp_path):
    staff = _write_staff_json(tmp_path / "s.json",
                              [["manually_corrected", "manually_grouped"]] * 3)
    assert is_manually_grouped(staff)


def test_plain_staff_finding_output_is_not_hand_grouped(tmp_path):
    staff = _write_staff_json(tmp_path / "s.json", [[], ["low_confidence"]])
    assert not is_manually_grouped(staff)


def test_entries_without_quality_are_not_hand_grouped(tmp_path):
    staff = tmp_path / "s.json"
    staff.write_text(json.dumps([{"id": "line0"}, {"id": "line1", "quality": None}]))
    assert not is_manually_grouped(staff)

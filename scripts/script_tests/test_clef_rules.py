"""Sanity checks for diatonic step <-> pitch conversion."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clef_rules import step_to_pitch


def test_zero_delta_returns_clef_pitch():
    assert step_to_pitch(0, "C", 4) == ("C", 4)


def test_positive_delta_steps_up_the_scale():
    assert step_to_pitch(1, "C", 4) == ("D", 4)
    assert step_to_pitch(6, "C", 4) == ("B", 4)


def test_delta_wraps_to_next_octave_up():
    assert step_to_pitch(7, "C", 4) == ("C", 5)
    assert step_to_pitch(8, "C", 4) == ("D", 5)


def test_negative_delta_steps_down_the_scale():
    assert step_to_pitch(-1, "C", 4) == ("B", 3)


def test_negative_delta_wraps_to_previous_octave():
    assert step_to_pitch(-7, "C", 4) == ("C", 3)


def test_fractional_delta_rounds_to_nearest_step():
    assert step_to_pitch(1.4, "C", 4) == ("D", 4)
    assert step_to_pitch(1.6, "C", 4) == ("E", 4)


def test_starting_from_a_non_c_clef():
    assert step_to_pitch(0, "F", 3) == ("F", 3)
    assert step_to_pitch(1, "F", 3) == ("G", 3)
    assert step_to_pitch(-1, "F", 3) == ("E", 3)

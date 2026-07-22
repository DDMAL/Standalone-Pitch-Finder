"""Sanity checks for the neume shape table, parsed from the real project CSV."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neume_shapes import load_neume_shapes

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "neumes-cheatsheet" / "csv-square_notation_neume_level_newest.csv"


def test_single_note_neume_has_only_interval_zero():
    shapes = load_neume_shapes(CSV_PATH)
    assert shapes.intervals_for("neume.punctum") == [0]
    assert shapes.intervals_for("neume.virga") == [0]


def test_two_note_neumes_have_correct_intervals():
    shapes = load_neume_shapes(CSV_PATH)
    print(f"clivis2: {shapes.intervals_for('neume.clivis2')}")
    assert shapes.intervals_for("neume.clivis2") == [0, -1]
    assert shapes.intervals_for("neume.podatus3") == [0, 2]


def test_three_note_neume_has_correct_intervals():
    shapes = load_neume_shapes(CSV_PATH)
    print(f"torculus22: {shapes.intervals_for('neume.torculus22')}")
    assert shapes.intervals_for("neume.torculus22") == [0, 1, -1]


def test_clef_classes_parsed():
    shapes = load_neume_shapes(CSV_PATH)
    assert shapes.is_clef("clef.c")
    assert shapes.clef_pname("clef.c") == "C"
    assert shapes.clef_pname("clef.f") == "F"


def test_custos_is_single_note():
    shapes = load_neume_shapes(CSV_PATH)
    assert shapes.intervals_for("custos") == [0]
    assert not shapes.is_pitchless("custos")


def test_pitchless_classes():
    shapes = load_neume_shapes(CSV_PATH)
    assert shapes.is_pitchless("divisio.maxima")
    assert shapes.is_pitchless("accidental.flat")
    assert shapes.is_pitchless("skip.dot")
    assert shapes.is_pitchless("skip.page")
    assert shapes.is_pitchless("text")


def test_unknown_class_has_no_intervals():
    shapes = load_neume_shapes(CSV_PATH)
    assert shapes.intervals_for("neume.totally_made_up") is None

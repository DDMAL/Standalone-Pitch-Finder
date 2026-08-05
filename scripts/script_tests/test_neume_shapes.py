"""Sanity checks for the neume shape table, parsed from the real project CSV."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neume_shapes import (load_neume_shapes, SOURCE_CSV, SOURCE_CSV_VARIANT,
                          SOURCE_BUILTIN, SOURCE_CLASS_NAME)

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "neumes-cheatsheet" / "csv-square_notation_neume_level_newest.csv"

# A multi-note class name and the digits in it, ignoring any trailing variant
# letter (podatus2b, scandicus22a) -- the letter distinguishes how the neume is
# drawn, never what it sounds like.
_NAME_RE = re.compile(r"^neume\.(?P<family>[a-z]+?)(?P<digits>\d+)[a-z]?$")


def intervals_from_class_name(class_name: str):
    """Diatonic offsets from the neume's first note, derived from its NAME.

    An oracle independent of the CSV: the classification names decode
    directly, so this can be checked against the MEI-derived table without
    sharing any code with it. `torculusAB` is up an Ath then down a Bth
    *from the preceding note*, `scandicusAB` up an Ath then up a Bth -- and an
    interval of an Nth spans N-1 diatonic steps.

    Returns None for names this scheme doesn't cover (single-note classes,
    custos, clefs), which the caller skips rather than asserting on.
    """
    m = _NAME_RE.match(class_name)
    if not m:
        return None
    family, digits = m.group("family"), m.group("digits")

    if family in ("podatus", "pescephalicus") and len(digits) == 1:
        return [0, int(digits) - 1]
    if family in ("clivis", "oblique") and len(digits) == 1:
        return [0, -(int(digits) - 1)]
    if family in ("torculus", "scandicus") and len(digits) == 2:
        a, b = int(digits[0]) - 1, int(digits[1]) - 1
        return [0, a, a - b] if family == "torculus" else [0, a, a + b]
    return None


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
    """@intm is relative to the PRECEDING note, so the offsets accumulate.

    torculus22 goes up a second and back down a second, ending where it
    started -- notes 1 and 3 share a pitch. Reading the CSV's `1S, -1S` as
    offsets from note 1 instead put note 3 a step below note 1, inverting the
    neume. Same for scandicus, which could not ascend at all: `1S, 1S` came
    out as [0, 1, 1], a repeated second note.
    """
    shapes = load_neume_shapes(CSV_PATH)
    print(f"torculus22: {shapes.intervals_for('neume.torculus22')}")
    assert shapes.intervals_for("neume.torculus22") == [0, 1, 0]
    assert shapes.intervals_for("neume.torculus33") == [0, 2, 0]
    assert shapes.intervals_for("neume.torculus42") == [0, 3, 2]
    assert shapes.intervals_for("neume.scandicus22a") == [0, 1, 2]
    assert shapes.intervals_for("neume.scandicus33") == [0, 2, 4]


def test_csv_intervals_match_intervals_derived_from_the_class_names():
    """Cross-check the whole table against an independent oracle.

    The classification names encode the same intervals the MEI skeletons do
    (torculusAB = up an Ath then down a Bth), so deriving them from the name
    reproduces the CSV exactly -- but only under the cumulative reading of
    @intm. This is the test that would have caught that bug, and it guards
    every multi-note class at once rather than the handful spelled out above.
    """
    shapes = load_neume_shapes(CSV_PATH)
    checked = {}
    for class_name, csv_intervals in shapes.neume_intervals.items():
        expected = intervals_from_class_name(class_name)
        if expected is None:
            continue
        checked[class_name] = csv_intervals
        assert csv_intervals == expected, (
            f"{class_name}: CSV says {csv_intervals}, name implies {expected}")

    print(f"cross-checked {len(checked)} classes: {checked}")
    # Guard the oracle itself: a regex that stopped matching would make this
    # test pass by checking nothing. Every multi-note class in the CSV is
    # covered by the name scheme.
    multi_note = {k for k, v in shapes.neume_intervals.items() if len(v) > 1}
    assert multi_note - set(checked) == set()
    assert len(checked) == len(multi_note) == 27


@pytest.mark.parametrize("class_name,expected", [
    ("neume.torculus22", [0, 1, 0]),      # up 2nd, down 2nd -> back to start
    ("neume.torculus23", [0, 1, -1]),     # descent outruns ascent -> ends lower
    ("neume.torculus42", [0, 3, 2]),      # ends above where it started
    ("neume.scandicus23", [0, 1, 3]),
    ("neume.podatus2b", [0, 1]),          # trailing variant letter ignored
    ("neume.clivis4", [0, -3]),
    ("neume.pescephalicus3", [0, 2]),
    ("neume.punctum", None),              # single-note: not name-derivable
    ("custos", None),
    ("clef.c", None),
])
def test_name_derived_intervals(class_name, expected):
    assert intervals_from_class_name(class_name) == expected


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
    assert shapes.intervals_with_source("neume.totally_made_up") == (None, None)


# --- classes real IC output emits that the CSV's 39 rows do not list --------

@pytest.mark.parametrize("class_name,base", [
    ("neume.clivis2a", "neume.clivis2"),
    ("neume.clivis2b", "neume.clivis2"),
    ("neume.clivis3a", "neume.clivis3"),
    ("neume.clivis4b", "neume.clivis4"),
    ("neume.podatus3b", "neume.podatus3"),
    ("neume.torculus22a", "neume.torculus22"),
])
def test_a_trailing_variant_letter_inherits_the_base_class_intervals(class_name, base):
    """The letter says which stroke carries the ligature, not what it sounds
    like -- so it is dropped for the intervals lookup. These are 32 glyphs on
    McGill_MS234-064 that used to lose every note but their first."""
    shapes = load_neume_shapes(CSV_PATH)
    intervals, source = shapes.intervals_with_source(class_name)
    assert intervals == shapes.intervals_for(base)
    assert source == SOURCE_CSV_VARIANT
    # ...and the base class itself still reports as CSV-backed, not a variant.
    assert shapes.intervals_with_source(base)[1] == SOURCE_CSV


@pytest.mark.parametrize("class_name,expected", [
    ("neume.podatus2", [0, 1]),
    ("neume.scandicus32", [0, 2, 3]),   # in the cheatsheet images, not the CSV
])
def test_classes_absent_from_the_csv_decode_from_their_names(class_name, expected):
    shapes = load_neume_shapes(CSV_PATH)
    assert class_name not in shapes.neume_intervals
    intervals, source = shapes.intervals_with_source(class_name)
    print(f"{class_name}: {intervals}")
    assert intervals == expected
    assert source == SOURCE_CLASS_NAME


@pytest.mark.parametrize("class_name", [
    "neume.distropha",       # 7 on McGill_MS234-064; in no CSV row
    "neume.tristropha",
    "neume.bistropha",
])
def test_repeated_note_neumes_resolve_to_a_single_pitch(class_name):
    """Every notehead of a distropha is on the same pitch.

    Decomposing it into N repeats adds no pitch information and hands
    downstream a unison nobody asked for, so it resolves to one note. It is a
    curated entry rather than a decode of the name -- there is no interval left
    to get wrong -- so unlike SOURCE_CLASS_NAME it carries no flag and the
    debug overlay draws it as a normal pitch.
    """
    shapes = load_neume_shapes(CSV_PATH)
    assert class_name not in shapes.neume_intervals
    assert shapes.intervals_with_source(class_name) == ([0], SOURCE_BUILTIN)
    assert not shapes.is_pitchless(class_name)


@pytest.mark.parametrize("class_name", [
    "neume.clivis1", "neume.podatus1", "neume.oblique1", "neume.pescephalicus1",
    "neume.torculus12", "neume.torculus21", "neume.scandicus11",
])
def test_a_unison_interval_is_a_misclassification_not_a_shape(class_name):
    """`clivisN` is a descent of an Nth, so `clivis1` is a descent of nothing.

    A ligature ties noteheads on *different* pitches; two on one pitch is a
    distropha. So a digit of 1 means the classifier was wrong, and decoding it
    to a unison would present that error as a confident two-note reading.
    These fall through to the flagged single-note path instead.
    """
    shapes = load_neume_shapes(CSV_PATH)
    assert shapes.intervals_with_source(class_name) == (None, None)


def test_name_fallback_agrees_with_the_independent_oracle():
    """The fallback generator and this file's oracle are separate code.

    intervals_from_class_name above is what cross-checks the CSV, so it has to
    stay independent of the module -- which leaves two implementations of the
    same decoding. This is what keeps them from drifting apart, over classes
    the CSV cannot check because it does not contain them.
    """
    shapes = load_neume_shapes(CSV_PATH)
    checked = {}
    # Digit 1 is excluded: the oracle decodes it (it is pure arithmetic) but
    # the module refuses it as a misclassification. That divergence is
    # deliberate and is pinned by its own test above.
    for family, digits in [("podatus", "2345"), ("clivis", "2345"),
                           ("oblique", "234"), ("pescephalicus", "23")]:
        for d in digits:
            checked[f"neume.{family}{d}x"] = None
    for a in "234":
        for b in "234":
            checked[f"neume.torculus{a}{b}"] = None
            checked[f"neume.scandicus{a}{b}"] = None

    for class_name in checked:
        expected = intervals_from_class_name(class_name)
        assert expected is not None, class_name
        assert shapes.intervals_with_source(class_name)[0] == expected, class_name
    print(f"cross-checked {len(checked)} name-derived classes")


def test_repeated_pitch_neumes_are_not_reachable_from_the_digit_scheme():
    """Guards the oracle used just above: distropha and friends carry no
    digits, so the name-derived path has to spell them out. If they ever start
    decoding from digits, the table below is dead code and should be deleted.
    """
    for class_name in ("neume.distropha", "neume.tristropha", "neume.bistropha"):
        assert intervals_from_class_name(class_name) is None

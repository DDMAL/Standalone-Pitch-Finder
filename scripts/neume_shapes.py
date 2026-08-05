"""
Neume shape table, parsed from the square-notation neume cheatsheet CSV.

That CSV (neumes-cheatsheet/csv-square_notation_neume_level_newest.csv) maps
every Gamera classification name to its MEI skeleton, and critically, each
note component's melodic interval is already encoded there as intm="1S" /
intm="-2S" (S = diatonic scale steps). This lets pitch_finder.py decompose a
multi-note neume glyph (podatus, clivis, torculus, scandicus, ...) into
individually-pitched note components instead of only reporting one anchor
pitch per glyph.

Classes with no intm attributes at all (single-note neumes like punctum,
virga) get an interval list of just [0].

The CSV enumerates 39 classifications; real IC output does not stay inside
that list, so lookup falls back twice before giving up (see
`intervals_with_source`): a trailing variant letter is stripped, then the
intervals are derived from the class name itself.
"""

import csv
import re
from pathlib import Path
from typing import Optional

_INTM_RE = re.compile(r'intm="(-?\d+)S"')

# Where a class's interval list came from. The caller flags anything that is
# not CSV-backed, so a derived shape is never mistaken for a documented one.
SOURCE_CSV = "csv"              # the classification is in the CSV verbatim
SOURCE_CSV_VARIANT = "csv_variant"  # ...after dropping a trailing variant letter
SOURCE_BUILTIN = "builtin"          # not in the CSV; a curated entry below
SOURCE_CLASS_NAME = "class_name"    # not in the CSV; decoded from the name

# "neume.clivis2a" -> "neume.clivis2". The trailing letter distinguishes how a
# neume is *drawn* (which stroke carries the ligature), never what it sounds
# like -- so it can be dropped for an intervals lookup but not for the pixel
# crop rules, which care about the drawing. script_tests/test_neume_shapes.py
# encodes the same assumption in its name-derived oracle.
_VARIANT_SUFFIX_RE = re.compile(r"^(neume\.[a-z]+\d+)[a-z]$")

# Multi-note classes decode straight from their names: `torculusAB` is up an
# Ath then down a Bth *from the preceding note*, `scandicusAB` up an Ath then
# up a Bth, and an interval of an Nth spans N-1 diatonic steps. This is the
# same scheme the CSV cross-check test derives independently, which is why it
# is safe to fall back on for the classes the CSV omits (neume.podatus2,
# neume.scandicus32, ...).
_NAME_SHAPE_RE = re.compile(r"^neume\.(?P<family>[a-z]+?)(?P<digits>\d+)[a-z]?$")

# Repeated-note neumes: N noteheads all on one pitch. The CSV has no row for
# them and they carry no digits, so neither lookup above reaches them -- yet
# neume.distropha alone appears 7 times on McGill_MS234-064.
#
# They resolve to ONE note, not N. Every notehead of a distropha carries the
# same pitch, so decomposing it into repeats adds no pitch information and
# leaves downstream consumers to collapse a unison they never asked for. This
# is a deliberate curated entry rather than a decode of the name, hence
# SOURCE_BUILTIN and no flag: there is no interval left to get wrong.
_BUILTIN_INTERVALS = {
    "neume.bistropha": [0],
    "neume.distropha": [0],
    "neume.tristropha": [0],
}

# Fallback for clef variants the CSV doesn't enumerate by exact name (e.g.
# "clef.f2", "clef.g" seen in real IC output) -- the letter right after
# "clef." is unambiguous regardless of any trailing variant suffix.
_CLEF_NAME_RE = re.compile(r'^clef\.([A-Za-z])')

# Hardcoded because they never appear in the CSV: skip.* is Gamera's junk
# catch-all (not a real musical symbol), and "text" is the placeholder name
# IC carries through for YOLO text-region bboxes that were never actually
# run through the neume classifier.
_ALWAYS_PITCHLESS_PREFIXES = ("skip.", "text")


def _extract_intervals(mei: str) -> list[int]:
    """Each note component's offset from the neume's FIRST note, in steps.

    MEI defines @intm on <nc> as the melodic interval from the *preceding*
    component, so the CSV's values are deltas and have to accumulate rather
    than being read as offsets from note 1. torculus22's `1S, -1S` is "up a
    second, then back down a second", i.e. [0, 1, 0] -- notes 1 and 3 on the
    same pitch, which is what the cheatsheet images show -- not [0, 1, -1].

    Reading them as absolute offsets inverted the contour of every neume that
    changes direction: it put a torculus's third note a full ascent below its
    first, and left a scandicus unable to ascend at all (`1S, 1S` came out as
    a repeated pitch). script_tests/test_neume_shapes.py cross-checks the
    whole table against intervals derived independently from the class names,
    which pins this permanently.
    """
    offsets = [0]
    for delta in _INTM_RE.findall(mei):
        offsets.append(offsets[-1] + int(delta))
    return offsets


def _intervals_from_class_name(class_name: str) -> Optional[list[int]]:
    """Each note component's offset from the neume's first note, decoded from
    the classification name alone. None if the name doesn't decode.

    Used only after both CSV lookups miss. It is the same decoding the CSV's
    own MEI encodes -- test_neume_shapes cross-checks all 27 multi-note CSV
    classes against an independent copy of it -- so extending it to the
    classes the CSV omits adds coverage, not a new convention.
    """
    if not class_name.startswith("neume."):
        return None

    m = _NAME_SHAPE_RE.match(class_name)
    if not m:
        return None
    family, digits = m.group("family"), m.group("digits")

    # A digit of 1 is an interval of a unison, and a ligature does not ligate
    # two noteheads on the same pitch -- that is what a distropha is. So
    # neume.clivis1 is not a neume, it is a misclassification, and decoding it
    # to [0, 0] would dress that up as a confident two-note reading. Refusing
    # sends it to pitch_finder's approximate_unknown_shape path instead, which
    # is flagged and drawn amber.
    if "1" in digits:
        return None

    if family in ("podatus", "pescephalicus") and len(digits) == 1:
        return [0, int(digits) - 1]
    if family in ("clivis", "oblique") and len(digits) == 1:
        return [0, -(int(digits) - 1)]
    if family in ("torculus", "scandicus") and len(digits) == 2:
        a, b = int(digits[0]) - 1, int(digits[1]) - 1
        return [0, a, a - b] if family == "torculus" else [0, a, a + b]
    return None


class NeumeShapeTable:
    """Holds the classification -> interval-list / clef / pitchless mappings."""

    def __init__(self, neume_intervals: dict[str, list[int]], clef_classes: dict[str, str],
                 pitchless_classes: set[str]):
        self.neume_intervals = neume_intervals
        self.clef_classes = clef_classes
        self.pitchless_classes = pitchless_classes

    def is_pitchless(self, class_name: str) -> bool:
        if class_name in self.pitchless_classes:
            return True
        return any(class_name.startswith(p) for p in _ALWAYS_PITCHLESS_PREFIXES)

    def is_clef(self, class_name: str) -> bool:
        return class_name in self.clef_classes or bool(_CLEF_NAME_RE.match(class_name))

    def clef_pname(self, class_name: str) -> str:
        if class_name in self.clef_classes:
            return self.clef_classes[class_name]
        m = _CLEF_NAME_RE.match(class_name)
        if not m:
            raise KeyError(f"{class_name!r} is not a clef class")
        return m.group(1).upper()

    def intervals_with_source(self, class_name: str) -> tuple[Optional[list[int]], Optional[str]]:
        """(interval list, SOURCE_*) for a classification, or (None, None).

        Three lookups, weakest evidence last. Only the exact CSV hit is
        documented ground truth, so the source is returned alongside and the
        caller flags anything below it:

        1. the CSV verbatim;
        2. the curated `_BUILTIN_INTERVALS` entries the CSV omits entirely
           (`neume.distropha` and the other repeated-note neumes);
        3. the CSV with a trailing variant letter dropped -- `neume.clivis2a`
           and `neume.clivis2b` are two ways of *drawing* a clivis2 and sound
           identical, so they inherit its `[0, -1]`;
        4. the class name decoded directly, for classes the CSV never lists
           (`neume.scandicus32`).

        Without 2-4, every one of those classes fell through to pitch_finder's
        single-note `approximate_unknown_shape` fallback and was flagged as a
        guess -- and the multi-note ones lost all but their first note. On
        McGill_MS234-064 that was 46 glyphs (31 clivis variants, 7 distropha, a
        podatus3b, ...), each reported as one note placed on the glyph's
        whole-shape ink centroid.
        """
        if self.is_clef(class_name):
            return [0], SOURCE_CSV

        exact = self.neume_intervals.get(class_name)
        if exact is not None:
            return exact, SOURCE_CSV

        builtin = _BUILTIN_INTERVALS.get(class_name)
        if builtin is not None:
            return list(builtin), SOURCE_BUILTIN

        m = _VARIANT_SUFFIX_RE.match(class_name)
        if m:
            base = self.neume_intervals.get(m.group(1))
            if base is not None:
                return base, SOURCE_CSV_VARIANT

        derived = _intervals_from_class_name(class_name)
        if derived is not None:
            return derived, SOURCE_CLASS_NAME

        return None, None

    def intervals_for(self, class_name: str) -> Optional[list[int]]:
        return self.intervals_with_source(class_name)[0]


def load_neume_shapes(csv_path: Path) -> NeumeShapeTable:
    neume_intervals: dict[str, list[int]] = {}
    clef_classes: dict[str, str] = {}
    pitchless_classes: set[str] = set()

    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_name = (row.get("classification") or "").strip()
            if not class_name:
                continue
            mei = row.get("mei") or ""

            if class_name.startswith("clef."):
                m = re.search(r'shape="([A-Za-z])"', mei)
                if m:
                    clef_classes[class_name] = m.group(1)
                continue

            if class_name == "custos":
                neume_intervals[class_name] = [0]
                continue

            if class_name.startswith("divisio.") or class_name.startswith("accidental."):
                pitchless_classes.add(class_name)
                continue

            if class_name.startswith("neume."):
                neume_intervals[class_name] = _extract_intervals(mei)
                continue

            # Unrecognized top-level category in the CSV: treat as pitchless
            # rather than silently dropping it.
            pitchless_classes.add(class_name)

    return NeumeShapeTable(neume_intervals, clef_classes, pitchless_classes)

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
"""

import csv
import re
from pathlib import Path
from typing import Optional

_INTM_RE = re.compile(r'intm="(-?\d+)S"')

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

    def intervals_for(self, class_name: str) -> Optional[list[int]]:
        if self.is_clef(class_name):
            return [0]
        return self.neume_intervals.get(class_name)


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

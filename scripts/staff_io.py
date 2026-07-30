"""
Staff-finding JSON I/O.

Parses the per-page staff-finding output (a list of fitted staff lines) into
Stave objects that pitch_finder.py can query for "what diatonic step is at
pixel (x, y)".

Step convention: within a stave, the bottom-most *detected* line is step 0;
each line index is 2 steps apart (the space between two adjacent lines is
step 1, 3, 5, ...). This mirrors standard staff notation (adjacent lines are
a third apart = 2 diatonic degrees).

The stave_id / within_stave_index in the file are staff-finding's own Stage 2
grouping, which groups by y alone and so merges the two columns of a
two-column page into single staves of eight lines. Loading therefore re-derives
the grouping from the line geometry by default (see staff_regroup); pass
regroup=False to take the file's grouping as-is. Either way the numbering
trusts that the lines of a stave were detected: if the *bottom* line of a stave
is missing, the step-0 anchor is off by whole steps. That is surfaced via the
sparse_stave_lines flag and the regrouping report, not fixed here.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

from staff_regroup import RegroupReport, regroup_entries


@dataclass
class StaffLine:
    """One fitted staff line, in page-pixel coordinates."""
    line_id: str
    stave_id: Optional[int]
    within_stave_index: Optional[int]
    x_start: int
    x_end: int
    y_values: list[float]
    scale_unit: float
    flags: list[str] = field(default_factory=list)

    def y_at_x(self, x: float) -> Optional[float]:
        """Interpolated y at page-pixel x, or None if x is outside this line's span."""
        if x < self.x_start or x > self.x_end:
            return None
        idx = round(x - self.x_start)
        idx = max(0, min(idx, len(self.y_values) - 1))
        return self.y_values[idx]


@dataclass
class Stave:
    """A stave: a group of StaffLines sharing a stave_id, sorted top-to-bottom."""
    stave_id: int
    lines: list[StaffLine]  # sorted by within_stave_index ascending (0 = topmost)

    def _max_within_index(self) -> int:
        return max(ln.within_stave_index for ln in self.lines)

    def step_at_x(self, x: float) -> Optional[list[tuple[float, float]]]:
        """(step, y) pairs at page-pixel x for every line of this stave that
        covers x, one pair per distinct step. None if no line covers x.

        Two lines can share a within_stave_index -- they are fragments of one
        physical line, split by an initial sitting in the stave -- and where
        their spans happen to overlap, both cover x. Collapsing them to a single
        averaged y keeps the steps distinct, which the interpolation below needs:
        two anchors at the same step define no slope, and silently return that
        step for every y.
        """
        max_idx = self._max_within_index()
        by_step: dict[float, list[float]] = {}
        for ln in self.lines:
            y = ln.y_at_x(x)
            if y is not None:
                by_step.setdefault(2 * (max_idx - ln.within_stave_index), []).append(y)
        pairs = [(step, sum(ys) / len(ys)) for step, ys in by_step.items()]
        return pairs or None

    def continuous_step_at_y(self, x: float, y: float) -> tuple[Optional[float], list[str]]:
        """Continuous (fractional) step value for an arbitrary y at page-pixel x.

        Linearly interpolates/extrapolates from the stave's known (step, y)
        points at that x. Returns (None, ["missing_staff"]) if the stave has
        no line coverage at x at all, and flags sparse_stave_lines when only
        one point is available (slope has to be guessed from scale_unit).
        """
        pairs = self.step_at_x(x)
        if not pairs:
            return None, ["missing_staff"]

        pairs = sorted(pairs, key=lambda p: p[1])  # sort by y ascending (top to bottom)
        if len(pairs) == 1:
            # Only one line available at this x: fall back to scale_unit as
            # the assumed pixel-per-step spacing so we can still extrapolate.
            step0, y0 = pairs[0]
            scale_unit = self.lines[0].scale_unit or 1.0
            half_gap = scale_unit / 2 if scale_unit else 1.0
            step = step0 - (y - y0) / half_gap
            return step, ["sparse_stave_lines"]

        # Use the two points nearest y for local linear interpolation/extrapolation.
        (step_top, y_top), (step_bottom, y_bottom) = pairs[0], pairs[-1]
        if y_bottom == y_top:
            return float(step_top), ["sparse_stave_lines"]
        t = (y - y_top) / (y_bottom - y_top)
        step = step_top + t * (step_bottom - step_top)
        return step, []

    def y_at_step(self, x: float, step: float) -> Optional[float]:
        """Page-pixel y of a (possibly fractional) step at page-pixel x.

        The exact inverse of continuous_step_at_y -- same two anchor points,
        same single-line scale_unit fallback -- so a step computed from a y
        maps back to that y. Pitch-finding works in steps, but a debug
        overlay has to draw in pixels; this is what converts a computed note
        position back into something drawable. None if the stave has no line
        coverage at x.
        """
        pairs = self.step_at_x(x)
        if not pairs:
            return None

        pairs = sorted(pairs, key=lambda p: p[1])  # sort by y ascending (top to bottom)
        if len(pairs) == 1:
            step0, y0 = pairs[0]
            scale_unit = self.lines[0].scale_unit or 1.0
            half_gap = scale_unit / 2 if scale_unit else 1.0
            return y0 - (step - step0) * half_gap

        (step_top, y_top), (step_bottom, y_bottom) = pairs[0], pairs[-1]
        if step_bottom == step_top:
            return float(y_top)
        t = (step - step_top) / (step_bottom - step_top)
        return y_top + t * (y_bottom - y_top)

    def nearest_line_distance(self, x: float, y: float) -> Optional[float]:
        """Vertical pixel distance from y to the nearest line of this stave at x."""
        pairs = self.step_at_x(x)
        if not pairs:
            return None
        return min(abs(y - ly) for _, ly in pairs)

    def y_span_at_x(self, x: float) -> Optional[tuple[float, float]]:
        """(min_y, max_y) across lines of this stave that cover x."""
        pairs = self.step_at_x(x)
        if not pairs:
            return None
        ys = [y for _, y in pairs]
        return min(ys), max(ys)

    def half_gap_at_x(self, x: float) -> float:
        """Local pixel spacing for one step, for converting a step-margin to pixels."""
        pairs = self.step_at_x(x)
        if pairs and len(pairs) >= 2:
            pairs = sorted(pairs, key=lambda p: p[1])
            step_span = pairs[-1][0] - pairs[0][0]
            y_span = pairs[-1][1] - pairs[0][1]
            if step_span:
                return abs(y_span / step_span)
        return (self.lines[0].scale_unit or 1.0) / 2


def load_staves(path: Path, *, regroup: bool = True) -> list[Stave]:
    """Parse a staff-finding JSON file into a list of Staves."""
    return load_staves_with_report(path, regroup=regroup)[0]


def load_staves_with_report(path: Path, *, regroup: bool = True
                            ) -> tuple[list[Stave], Optional[RegroupReport]]:
    """Parse a staff-finding JSON file into (Staves, regrouping report).

    With regroup=True (the default) the file's stave_id / within_stave_index are
    re-derived from the line geometry, and the report says what that did -- it is
    the only place a caller can see that, say, a page turned out to have two
    columns, so the runners print it. The report is None with regroup=False.

    Lines with stave_id or within_stave_index missing (staff-finding couldn't
    group them) are skipped -- they can't be assigned a step position, so they're
    not useful to pitch-finding. Regrouping assigns all of them, so this only
    drops lines when regrouping is off.
    """
    data = json.loads(Path(path).read_text())
    report = None
    if regroup:
        data, report = regroup_entries(data)

    by_stave: dict[int, list[StaffLine]] = {}
    for entry in data:
        stave_id = entry.get("stave_id")
        within_idx = entry.get("within_stave_index")
        if stave_id is None or within_idx is None:
            continue
        centerline_page = entry["centerline_page"]
        line = StaffLine(
            line_id=entry["id"],
            stave_id=stave_id,
            within_stave_index=within_idx,
            x_start=centerline_page["x_start"],
            x_end=centerline_page["x_end"],
            y_values=centerline_page["y_values"],
            scale_unit=entry.get("scale_unit"),
            flags=list(entry.get("quality", {}).get("flags", [])),
        )
        by_stave.setdefault(stave_id, []).append(line)

    staves = []
    for stave_id, lines in by_stave.items():
        lines.sort(key=lambda ln: ln.within_stave_index)
        staves.append(Stave(stave_id=stave_id, lines=lines))
    staves.sort(key=lambda s: s.stave_id)
    return staves, report

"""Re-derive stave grouping from staff-finding's raw line geometry.

Staff-finding tags every fitted line with a stave_id, but its Stage 2 grouping
is purely vertical: lines at a similar y become one stave. On a two-column page
that is wrong in the worst possible way -- the left column's four lines and the
right column's four lines sit at the same y, so they merge into one "stave" of
eight lines, and staff_io's `2 * (max_index - index)` numbering then reads that
as a 14-step staff instead of two 6-step ones. Every pitch on such a page is
wrong, and nothing downstream can tell.

This module throws that grouping away and rebuilds it from the geometry the
same file already carries (each line's x span and centerline y), in three steps:

1. **Columns.** Project every line's x span onto the x axis and count how many
   lines cover each pixel. A column gutter is a wide interior band that almost
   nothing covers. Counting coverage rather than just looking for uncovered
   x is what keeps a single full-width line -- a page border, a rubric rule --
   from bridging the two columns and hiding the gutter. Lines are assigned to
   the block containing their span's midpoint.

   Coverage counting is also what tells a gutter apart from the gap a decorated
   initial leaves *inside* a stave: nothing on the page crosses a gutter, whereas
   other staves cross an initial. So the assumption is that some stave bridges
   every such gap -- true on every page here, but a crop holding one interrupted
   stave and nothing else would have that stave split in two.

2. **Staves.** Within a column, sort by y and cut wherever the step to the next
   line exceeds STAVE_CUT_GAPS line gaps. The line gap is measured from the
   page itself (see `estimate_line_gap`), not from scale_unit, which does not
   track it: 16.5 goes with a 13.9px gap on one page here and 25.0 with 41.8 on
   another.

3. **Lines within a stave.** A physical staff line is often detected as several
   entries: a near-duplicate of itself, or two fragments split by a decorated
   initial sitting in the middle of the stave. Fragments of one line are ~1-4px
   apart while adjacent lines are a full gap apart, so each entry's
   within_stave_index is `round((y - y_top) / gap)` rather than its rank. That
   collapses fragments onto one index, and -- the reason it is worth doing even
   on a one-column page -- it leaves a hole at the index of a line that was
   never detected, instead of silently renumbering the lines below it and
   shifting every pitch on the stave by a step.

Near-duplicate entries (same line, overlapping x) are dropped; genuine
fragments (same line, disjoint x) are all kept, since between them they cover
more of the stave than either does alone.

An entry with no y_values at all (a degenerate zero-width centerline -- seen
from a detection too thin/spurious for staff-finding's own fit step to
produce any samples for) is dropped before any of the above: every grouping
step below reads a y from each entry, and one with nothing to average would
otherwise fail outright rather than just being geometry staff-finding
couldn't use. Such an entry already carries no stave_id of its own for the
same reason, so dropping it here changes nothing for a caller that was going
to discard it anyway once ungrouped.
"""

from dataclasses import dataclass, field
import math
import statistics

# A gutter must be at least this wide (page pixels) to split columns, and be
# covered by no more than this fraction of the page's peak line coverage.
MIN_GUTTER_PX = 20
GUTTER_COVERAGE_FRAC = 0.15

# Start a new stave when the vertical step to the next line exceeds this many
# line gaps. Measured on this repo's pages, the widest gap *inside* a stave
# (one line of four undetected) is 1.97 gaps and the narrowest gap *between*
# staves is 2.13, so there is not much room either side of 2.0.
STAVE_CUT_GAPS = 2.0

# Two lines must overlap in x by this fraction of the shorter one before their
# vertical distance is taken as a line-gap measurement (`estimate_line_gap`),
# or before they count as duplicates of one line rather than fragments of it.
MIN_X_OVERLAP_FRAC = 0.35
DUPLICATE_X_OVERLAP_FRAC = 0.5

# Line-gap multiples bounding what counts as a single-line step when refining
# the gap estimate inside one stave.
LOCAL_GAP_BOUNDS = (0.6, 1.5)


@dataclass
class RegroupReport:
    """What regrouping did to one page, for the runners to print."""
    lines_in: int
    lines_kept: int
    columns: int
    staves: int
    line_gap: float
    # stave line count -> how many staves have it, e.g. {4: 16, 3: 3, 1: 1}
    line_counts: dict[int, int] = field(default_factory=dict)
    incomplete: list[tuple[int, list[int]]] = field(default_factory=list)

    def summary(self) -> str:
        counts = ", ".join(f"{n} stave(s) with {k} line(s)"
                           for k, n in sorted(self.line_counts.items(), reverse=True))
        text = (f"Regrouped staff lines: {self.lines_in} line(s) in, "
                f"{self.lines_kept} kept, {self.columns} column(s), "
                f"{self.staves} stave(s), line gap {self.line_gap:.1f}px "
                f"({counts}).")
        if self.incomplete:
            detail = "; ".join(f"stave {sid}: lines {idxs}" for sid, idxs in self.incomplete)
            text += (f"\n  {len(self.incomplete)} stave(s) missing a detected line "
                     f"-- pitches there rest on fewer anchors ({detail}).")
        return text


def _x_span(entry: dict) -> tuple[int, int]:
    cp = entry["centerline_page"]
    return cp["x_start"], cp["x_end"]


def _y_mean(entry: dict) -> float:
    return statistics.fmean(entry["centerline_page"]["y_values"])


def _x_overlap(a: dict, b: dict) -> tuple[int, int]:
    """(overlapping x pixels, width of the shorter span) for two lines."""
    a0, a1 = _x_span(a)
    b0, b1 = _x_span(b)
    return min(a1, b1) - max(a0, b0), min(a1 - a0, b1 - b0)


def _neighbour_gaps(entries: list[dict], min_distance: float) -> list[float]:
    """Each line's distance to its nearest x-overlapping neighbour.

    Restricting to lines that overlap in x is the whole point: on a page where
    every line is detected as a left and a right fragment, the two fragments of
    one line interleave in a global y sort with the fragments of the next line,
    and plain successive differences measure the fragment offset (~4px) instead
    of the line gap (~12px). Fragments of the same line do not overlap in x;
    consecutive lines of the same fragment column do.

    min_distance skips near-duplicate detections of the same line, which would
    otherwise report a gap of ~0.
    """
    out = []
    for e in entries:
        best = None
        for f in entries:
            if f is e:
                continue
            overlap, shorter = _x_overlap(e, f)
            if shorter <= 0 or overlap < MIN_X_OVERLAP_FRAC * shorter:
                continue
            distance = abs(_y_mean(e) - _y_mean(f))
            if distance > min_distance and (best is None or distance < best):
                best = distance
        if best is not None:
            out.append(best)
    return out


def estimate_line_gap(entries: list[dict]) -> float | None:
    """Median vertical distance between adjacent staff lines on the page.

    Bootstrapped: a first pass rejects only distances under a pixel (exact
    duplicate detections), and each later pass uses a quarter of the previous
    estimate as the duplicate threshold, which is scale-free -- these pages
    range from a 13.9px line gap to a 41.8px one.
    """
    gap, min_distance = None, 1.0
    for _ in range(3):
        gaps = _neighbour_gaps(entries, min_distance)
        if not gaps:
            return gap
        gap = statistics.median(gaps)
        min_distance = 0.25 * gap
    return gap


def split_columns(entries: list[dict]) -> list[tuple[tuple[int, int], list[dict]]]:
    """Split lines into ((x_lo, x_hi), lines) column blocks. One block if the
    page is single-column."""
    lo = min(x0 for x0, _ in map(_x_span, entries))
    hi = max(x1 for _, x1 in map(_x_span, entries))
    coverage = [0] * (hi - lo + 1)
    for entry in entries:
        x0, x1 = _x_span(entry)
        for i in range(x0 - lo, x1 - lo + 1):
            coverage[i] += 1

    threshold = max(1, max(coverage) * GUTTER_COVERAGE_FRAC)
    gutters, run = [], None
    for i, count in enumerate(coverage):
        if count <= threshold:
            run = (run[0], i) if run else (i, i)
            continue
        if run and run[1] - run[0] + 1 >= MIN_GUTTER_PX:
            gutters.append(run)
        run = None
    # A trailing low-coverage run is the page margin, not a gutter, so it is
    # deliberately not flushed here.

    bounds = [lo]
    for g0, g1 in gutters:
        bounds += [lo + g0, lo + g1]
    bounds.append(hi)
    blocks = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds) - 1, 2)]

    columns = [[] for _ in blocks]
    for entry in entries:
        x0, x1 = _x_span(entry)
        mid = (x0 + x1) / 2
        best = min(range(len(blocks)),
                   key=lambda i: 0 if blocks[i][0] <= mid <= blocks[i][1]
                   else min(abs(mid - blocks[i][0]), abs(mid - blocks[i][1])))
        columns[best].append(entry)
    return [(block, lines) for block, lines in zip(blocks, columns) if lines]


def _split_staves(column: list[dict], gap: float) -> list[list[dict]]:
    """Cut one column's lines into staves at vertical steps wider than
    STAVE_CUT_GAPS line gaps."""
    staves = []
    for entry in sorted(column, key=_y_mean):
        if staves and _y_mean(entry) - _y_mean(staves[-1][-1]) > gap * STAVE_CUT_GAPS:
            staves.append([entry])
        elif staves:
            staves[-1].append(entry)
        else:
            staves.append([entry])
    return staves


def _local_gap(stave: list[dict], gap: float) -> float:
    """Refine the line gap using only this stave's lines.

    Staves drift in spacing across a warped or cropped page, and the index
    rounding below is a division by this number, so the page-wide median is
    worth localising. Only distances that already look like a single-line step
    are used -- otherwise a stave with an undetected line would measure its
    double gap as the unit.
    """
    lo, hi = LOCAL_GAP_BOUNDS
    candidates = [d for d in _neighbour_gaps(stave, 0.25 * gap)
                  if lo * gap <= d <= hi * gap]
    return statistics.median(candidates) if candidates else gap


def _line_indices(stave: list[dict], gap: float) -> list[tuple[int, float]]:
    """(within_stave_index, |rounding residual|) for each line of one stave.

    The index is the line's offset from the topmost line in gaps, rounded, so
    fragments of one physical line land on one index and an undetected line
    leaves its index unused. The topmost line is only a provisional origin: a
    spurious detection just above the stave would put every real line on a
    fractional offset, so the offset that best explains all of them is
    subtracted first, as the circular mean of the fractional parts (circular
    because these are positions modulo one gap -- 0.98 and 0.02 are a tenth of
    a gap apart, not most of one).
    """
    top = min(_y_mean(e) for e in stave)
    offsets = [(_y_mean(e) - top) / gap for e in stave]
    angles = [2 * math.pi * (o - math.floor(o)) for o in offsets]
    phase = math.atan2(statistics.fmean(map(math.sin, angles)),
                       statistics.fmean(map(math.cos, angles))) / (2 * math.pi)
    indices = [round(o - phase) for o in offsets]
    lowest = min(indices)
    return [(i - lowest, abs(o - phase - i)) for i, o in zip(indices, offsets)]


def _drop_duplicates(line: list[tuple[dict, float]]) -> list[dict]:
    """Keep one entry per x region of one physical line.

    Entries that overlap in x are competing detections of the same stretch of
    line; the best is the one closest to the stave's line spacing, then the one
    covering the most x, then the better pixel fit. The residual is bucketed so
    that true duplicates -- whose residuals differ by a hundredth of a step --
    are settled on span rather than on noise. Entries that do not overlap are
    fragments either side of an initial, and all of them are kept.
    """
    def rank(item):
        entry, residual = item
        x0, x1 = _x_span(entry)
        return (round(residual / 0.05), -(x1 - x0),
                entry.get("fit", {}).get("residual_mean") or 0.0)

    kept: list[dict] = []
    for entry, _ in sorted(line, key=rank):
        if not any(_x_overlap(entry, k)[0] > DUPLICATE_X_OVERLAP_FRAC * _x_overlap(entry, k)[1]
                   for k in kept):
            kept.append(entry)
    return kept


def regroup_entries(entries: list[dict]) -> tuple[list[dict], RegroupReport]:
    """Rewrite column_id / stave_id / within_stave_index from line geometry.

    Returns (entries, report). Entries are shallow copies with those three
    fields replaced -- the input list is not modified -- and near-duplicate
    detections are dropped. Ordering is by column, then stave, then line.
    """
    if not entries:
        return [], RegroupReport(0, 0, 0, 0, 0.0)

    lines_in = len(entries)
    entries = [e for e in entries if e["centerline_page"]["y_values"]]
    if not entries:
        return [], RegroupReport(lines_in, 0, 0, 0, 0.0)

    gap = estimate_line_gap(entries) or 1.0
    columns = split_columns(entries)

    out: list[dict] = []
    counts: dict[int, int] = {}
    incomplete: list[tuple[int, list[int]]] = []
    stave_id = 0
    for column_id, (_block, column) in enumerate(columns):
        for stave in _split_staves(column, gap):
            local = _local_gap(stave, gap)
            by_line: dict[int, list[tuple[dict, float]]] = {}
            for (index, residual), entry in zip(_line_indices(stave, local), stave):
                by_line.setdefault(index, []).append((entry, residual))

            indices = sorted(by_line)
            for index in indices:
                for entry in _drop_duplicates(by_line[index]):
                    out.append({**entry, "column_id": column_id,
                                "stave_id": stave_id, "within_stave_index": index})
            counts[len(indices)] = counts.get(len(indices), 0) + 1
            if indices != list(range(len(indices))):
                incomplete.append((stave_id, indices))
            stave_id += 1

    report = RegroupReport(lines_in=lines_in, lines_kept=len(out),
                           columns=len(columns), staves=stave_id, line_gap=gap,
                           line_counts=counts, incomplete=incomplete)
    return out, report

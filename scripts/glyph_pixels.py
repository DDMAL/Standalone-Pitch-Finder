"""
Pixel-level glyph analysis for the Rodan-style pitch finder.

Ports the pixel-analysis half of Rodan's heuristic pitch finder --
_x_projection_vector / get_subimage / _center_of_mass / _vector_process_f_clef
in https://github.com/DDMAL/Rodan/blob/master/rodan-main/code/rodan/jobs/heuristic_pitch_finding/PitchFinding.py
-- without needing Gamera: we crop straight from our own manuscript image
using the IC bbox, instead of decoding Gamera's onebit image encoding.

Rodan's rationale (kept here): a glyph's bounding box is not a reliable
pitch reference on its own -- a virga's descending stem, or a podatus's
second notehead, pull the bbox's geometric center away from the actual
notehead. So for a handful of classes, only a sub-region of the bbox is
used for the row-projection centroid:
  - virga: top region only (the stem trails below the notehead)
  - podatus2b/3/4/5, scandicus22b: bottom-left region only (these are
    drawn ligated with the reference notehead at the bottom-left). On the
    decomposition path this widens to every ascending ligature, the mirror
    of it covers the descending clivis, and both locate the region by ink
    rather than by a bbox edge -- see FIRST_HEAD_INK_CLASSES /
    TOP_HEAD_INK_CLASSES / _head_ink_region.
  - clef.f* (any F-clef variant): right half only, trimmed to its own ink
    extent (the two dots on an F-clef's left side aren't the reference point)
  - everything else: full height, width capped at the average punctum size
    (so an unusually wide multi-note glyph doesn't get diluted by grabbing
    a neighboring symbol)

Two entry points read that region:
  - reference_row: the row offset alone, for rodan_pitch_finder's
    one-pitch-per-glyph flow (unchanged, still Rodan-faithful).
  - reference_point: the full (x, y, region) point, for pitch_finder's
    multi-note decomposition. It needs the region label because *which*
    notehead the centroid represents depends on which crop produced it --
    see REGION_* below.

Only the decomposition path adds crop rules of its own (extended_rules: the
first-head rules, FIRST_HEAD_INK_CLASSES and TOP_HEAD_INK_CLASSES). Rodan
has one pitch per glyph and can settle for an ink centroid that belongs to
no particular notehead; a caller that places three notes off that one point
cannot. Keeping those rules off the reference_row path is what leaves
rodan_pitch_finder an independent baseline instead of a copy of this
module's opinions.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

from ic_io import Glyph

BOTTOM_LEFT_CLASSES = {"neume.podatus2b", "neume.podatus3", "neume.podatus4",
                        "neume.podatus5", "neume.scandicus22b"}

# Classes cropped to their first notehead on the decomposition path only (see
# the module docstring's note on extended_rules). Every one of them is an
# ascending ligature drawn from a bottom-left head, so "the first note" and
# "the lowest head in the left column band" are the same thing.
#
# Rodan crops podatus2b/3/4/5 and scandicus22b to a band hanging off the bbox
# bottom, and has no torculus rule at all. Both are wrong for a caller that
# places several notes off the one point -- see _first_head_ink_region.
FIRST_HEAD_INK_CLASSES = ("neume.podatus", "neume.pescephalicus",
                          "neume.scandicus", "neume.torculus")

# The mirror of the above, for the descending ligature: a clivis is drawn from
# its higher head down to its lower one, so the first note is the TOP head of
# the left column band. `neume.oblique*` is deliberately absent -- it is a
# solid diagonal parallelogram with no separable heads, so its whole-band
# centroid genuinely is the midpoint of its two notes, and forcing this rule on
# it takes `oblique3`'s second note from 0.35 steps of error to 0.95.
TOP_HEAD_INK_CLASSES = ("neume.clivis",)

# How deep the top-head crop reaches, as a fraction of one notehead.
#
# Unlike the bottom-head crop, this cap is load-bearing rather than a safety
# net, which is why it is tighter than a whole notehead. A podatus keeps its
# ligature stroke on the *right*, outside the left band, so the band's lowest
# run is the bare notehead -- a median 0.68 noteheads tall, never over 1.3, so
# the cap almost never engages. A clivis's left stem descends *inside* the
# band, fused to the head: on McGill_MS234-064 that run runs 1.68 noteheads,
# 64% of them past 1.3. The cap is then the only thing separating head from
# stem, and every pixel of it moves the anchor.
#
# 0.75 is the middle of a plateau: 0.70-0.75 both leave 92% of clivis second
# notes inside half a step (against 88% at a full notehead) with a signed bias
# of -0.01 steps, i.e. unbiased.
TOP_HEAD_DEPTH_FRACTION = 0.75

# A row run must carry this share of the heaviest run's ink to count as a
# notehead. Staff lines cross the column band and would otherwise read as
# noteheads of their own; they are 2-3 rows against a head's 15-25, so an
# order of magnitude separates them and the exact cutoff barely matters
# (0.45-0.65 all score within a hair on both sample pages).
HEAD_MASS_FRACTION = 0.50

# Rows below the peak at which the band is considered to have stopped. Held
# well below the peak so a head thinning towards its edge stays one run.
_INK_RUN_THRESHOLD = 0.20

# Which sub-region of the bbox the centroid was measured from. Rodan itself
# never needs this (one pitch per glyph, so the point is simply "the" pitch),
# but a caller that decomposes a neume into several notes does: a bottom-left
# crop lands on the neume's FIRST notehead, a top crop on its HIGHEST, and a
# full-bbox crop is the ink centroid of the whole shape and so belongs to no
# single note. Getting that wrong offsets every note of the neume at once.
REGION_FULL = "full"
REGION_TOP = "top"
REGION_BOTTOM_LEFT = "bottom_left"
REGION_F_CLEF_RIGHT = "f_clef_right"


@dataclass
class ReferenceRegion:
    """The page-pixel rectangle a glyph's pitch reference is measured from."""
    ulx: int
    uly: int
    ncols: int
    nrows: int
    region: str


@dataclass
class ReferencePoint:
    """A measured pitch reference: page-pixel point + the crop it came from."""
    x: float
    y: float
    region: str


def average_punctum(glyphs: list[Glyph]) -> float:
    """Average bbox width of punctum/virga glyphs -- Rodan's proxy for
    'one notehead's width', used to size the special-case crops below."""
    widths = [g.ncols for g in glyphs if g.class_name in ("neume.punctum", "neume.virga")]
    return sum(widths) / len(widths) if widths else 0.0


def notehead_height(glyphs: list[Glyph]) -> float:
    """Median bbox height of the page's single-note glyphs -- how deep one
    notehead is, in this page's own pixels. 0.0 if the page has none.

    Rodan sizes its crops from average_punctum and uses that one number as
    both a width and a height. It is a width: on McGill_MS234-064 it is 35 px
    while a notehead is 25 px tall, so a band "one notehead deep" came out
    40% too deep and reached into the ligature stroke above the head. virga is
    excluded (unlike average_punctum) precisely because its bbox includes the
    stem, which is the thing being measured around.
    """
    heights = sorted(g.nrows for g in glyphs
                     if g.class_name in ("neume.punctum", "neume.inclinatum"))
    if not heights:
        return 0.0
    mid = len(heights) // 2
    if len(heights) % 2:
        return float(heights[mid])
    return (heights[mid - 1] + heights[mid]) / 2


def crop_and_binarize(image: np.ndarray, ulx: float, uly: float, ncols: float, nrows: float) -> np.ndarray:
    """Crop a region from the page image and Otsu-binarize it (ink = 255)."""
    ulx, uly, ncols, nrows = round(ulx), round(uly), round(ncols), round(nrows)
    h, w = image.shape[:2]
    x0, y0 = max(0, ulx), max(0, uly)
    x1, y1 = min(w, ulx + ncols), min(h, uly + nrows)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 0), dtype=np.uint8)

    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def row_projection_centroid(binary_crop: np.ndarray) -> float:
    """Ink-weighted row centroid, 1-indexed (row 1 = the crop's top row).

    Matches Rodan's _center_of_mass exactly: s = sum((i+1)*count_i),
    v = sum(count_i), com = s/v (0 if there's no ink at all).
    """
    if binary_crop.size == 0:
        return 0.0
    projection = (binary_crop != 0).sum(axis=1).astype(float)
    total = projection.sum()
    if total == 0:
        return 0.0
    weights = np.arange(1, len(projection) + 1, dtype=float)
    return float((weights * projection).sum() / total)


def _f_clef_right_region(image: np.ndarray, glyph: Glyph) -> tuple:
    """Approximates _vector_process_f_clef: use the right half of the bbox,
    trimmed to its own ink extent, instead of Gamera's connected-component
    union (which needs real Gamera objects we don't have).

    Returns (ulx, uly, ncols, nrows, y_add) for the trimmed right-half region.
    y_add is the region's top offset relative to the original glyph top.
    """
    right_ulx = glyph.ulx + glyph.ncols // 2
    right_ncols = glyph.ulx + glyph.ncols - right_ulx
    crop = crop_and_binarize(image, right_ulx, glyph.uly, right_ncols, glyph.nrows)
    rows_with_ink = np.where((crop != 0).any(axis=1))[0]
    if len(rows_with_ink) == 0:
        return right_ulx, glyph.uly, right_ncols, glyph.nrows, 0.0
    top, bottom = int(rows_with_ink[0]), int(rows_with_ink[-1])
    return right_ulx, glyph.uly + top, right_ncols, bottom - top + 1, float(top)


def _ink_runs(projection: np.ndarray) -> list[tuple]:
    """Contiguous [start, stop) row ranges where the band carries ink."""
    on = projection > _INK_RUN_THRESHOLD * projection.max()
    runs, start = [], None
    for i, inked in enumerate(on):
        if inked and start is None:
            start = i
        elif not inked and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(on)))
    return runs


def _notehead_runs(projection: np.ndarray) -> list[tuple]:
    """The ink runs heavy enough to be noteheads rather than staff lines.

    A staff line crossing the column band inks the band's full width, so it
    cannot be told from a notehead row by row -- only by how many rows it
    spans. Weighing each run by its total ink separates them by roughly an
    order of magnitude (2-3 rows against 15-25), and does it in the page's own
    units, with no threshold to carry between manuscripts.
    """
    runs = _ink_runs(projection)
    if not runs:
        return []
    masses = [projection[a:b].sum() for a, b in runs]
    cutoff = HEAD_MASS_FRACTION * max(masses)
    return [run for run, mass in zip(runs, masses) if mass >= cutoff]


def _head_ink_region(image: np.ndarray, glyph: Glyph, extend_cols: int,
                      head_depth: int, *, from_bottom: bool,
                      region: str) -> ReferenceRegion:
    """The neume's first notehead: the outermost notehead-sized run of ink in
    the left column band, rather than a band hanging off a bbox edge.

    from_bottom picks which end of the band the first note is drawn at -- the
    bottom for an ascending ligature (podatus, torculus), the top for a
    descending one (clivis). Either way the run that survives *is* the first
    note, so the caller binds it to interval 0 with nothing to derive.

    Two things go wrong with a bbox-anchored band, and both put the anchor
    on the wrong side of the head it is supposed to be measuring:

    - **The bbox bottom is not the first head's bottom.** It is set by
      whatever ink hangs lowest, which is the ligature's descending right-hand
      stroke on a podatus, and the *third* note on a torculus whose descent
      outruns its ascent (torculus23/24/34, e.g. [0, 1, -2]). The band slides
      down past the head and clips its top off, biasing the centroid low --
      measured at 0.27 steps on McGill_MS234-064's podatus2b, up to 0.96 on
      the worst one, against a rounding threshold of 0.5.
      The mirror holds for a clivis and its bbox *top*, and is worse there:
      the midpoint anchor it replaced put `clivis4b`'s first note 1.15 steps
      too high -- outside the glyph's own bbox.
    - **Staff lines are ink too.** They cross the band as thin runs and drag
      any centroid computed over a fixed window towards themselves.

    Segmenting the band's row profile and keeping only the notehead-sized runs
    (_notehead_runs) answers both: the outermost surviving run *is* the first
    head, wherever the bbox happens to end. It is then clamped to head_depth
    so that a head fused to its own stroke -- one run spanning the glyph --
    still contributes only its own notehead's worth of rows.

    A band with no ink at all is handed back untrimmed, so the caller's own
    no-ink path decides what to do about it.
    """
    band = crop_and_binarize(image, glyph.ulx, glyph.uly, extend_cols, glyph.nrows)
    full = ReferenceRegion(glyph.ulx, glyph.uly, extend_cols, glyph.nrows, region)
    if band.size == 0:
        return full
    projection = (band != 0).sum(axis=1).astype(float)
    if projection.max() == 0:
        return full

    runs = _notehead_runs(projection)
    if not runs:
        return full
    if from_bottom:
        top, bottom = runs[-1]
        top = max(top, bottom - head_depth)
    else:
        top, bottom = runs[0]
        bottom = min(bottom, top + head_depth)
    return ReferenceRegion(glyph.ulx, glyph.uly + top, extend_cols,
                           bottom - top, region)


def reference_region(image: np.ndarray, glyph: Glyph, avg_punctum: float,
                      discard_size: int = 12,
                      subimage_width_factor: float = 0.8, *,
                      extended_rules: bool = False,
                      notehead_h: float = 0.0) -> Optional[ReferenceRegion]:
    """The sub-region of the glyph's bbox whose ink centroid is the pitch
    reference -- the per-class crop rules described in the module docstring.

    extended_rules adds the crop rules Rodan does not have (FIRST_HEAD_INK_CLASSES);
    it defaults off so this stays Rodan-faithful for reference_row's caller, and
    reference_point turns it on.

    notehead_h (see notehead_height) sets how deep one notehead is, and is only
    read under extended_rules. Left at 0 the rule falls back to avg_punctum,
    which is a notehead's *width* -- workable, just looser.

    None for glyphs too small for pixel analysis to mean anything (both dims
    <= discard_size), which is Rodan's discard_size behavior.
    """
    if glyph.ncols <= discard_size and glyph.nrows <= discard_size:
        return None

    ulx, uly, ncols, nrows = glyph.ulx, glyph.uly, glyph.ncols, glyph.nrows
    region = REGION_FULL

    if glyph.class_name.startswith("clef.f"):
        ulx, uly, ncols, nrows, _y_add = _f_clef_right_region(image, glyph)
        region = REGION_F_CLEF_RIGHT

    extend_cols = ncols if ncols < avg_punctum else avg_punctum * subimage_width_factor
    extend_rows = nrows if nrows < avg_punctum else avg_punctum
    extend_cols = max(1, round(extend_cols))
    extend_rows = max(1, round(extend_rows))

    if extended_rules and glyph.class_name.startswith(FIRST_HEAD_INK_CLASSES):
        head_depth = min(notehead_h or extend_rows, glyph.nrows)
        return _head_ink_region(image, glyph, extend_cols,
                                max(1, round(head_depth)), from_bottom=True,
                                region=REGION_BOTTOM_LEFT)

    if extended_rules and glyph.class_name.startswith(TOP_HEAD_INK_CLASSES):
        # REGION_TOP, so _anchor_interval binds this to max(intervals) -- and a
        # clivis descends, so its highest note is also its first.
        head_depth = min((notehead_h or extend_rows) * TOP_HEAD_DEPTH_FRACTION,
                         glyph.nrows)
        return _head_ink_region(image, glyph, extend_cols,
                                max(1, round(head_depth)), from_bottom=False,
                                region=REGION_TOP)

    if glyph.class_name in BOTTOM_LEFT_CLASSES:
        return ReferenceRegion(ulx, uly + nrows - extend_rows, extend_cols,
                               extend_rows, REGION_BOTTOM_LEFT)

    if glyph.class_name == "neume.virga":
        return ReferenceRegion(ulx, uly, extend_cols, extend_rows, REGION_TOP)

    # Default: full height, width capped to avoid grabbing a neighbor.
    return ReferenceRegion(ulx, uly, extend_cols, nrows, region)


def reference_row(image: np.ndarray, glyph: Glyph, avg_punctum: float,
                   discard_size: int = 12, subimage_width_factor: float = 0.8) -> float:
    """The pitch-reference row, as an offset from the glyph's own top edge
    (uly). Add to glyph.uly to get a page-pixel y coordinate.

    Tiny glyphs (both dims <= discard_size) skip pixel analysis entirely and
    get 0.0, matching Rodan's center_of_mass=0 fallback for glyphs too small
    to meaningfully binarize/project. A crop with no ink at all also
    centroids to 0.0 (row_projection_centroid) -- indistinguishable from a
    real measurement at the top edge, again as in Rodan. Callers that need to
    tell those apart should use reference_point instead.
    """
    region = reference_region(image, glyph, avg_punctum, discard_size, subimage_width_factor)
    if region is None:
        return 0.0
    crop = crop_and_binarize(image, region.ulx, region.uly, region.ncols, region.nrows)
    return (region.uly - glyph.uly) + row_projection_centroid(crop)


def reference_point(image: np.ndarray, glyph: Glyph, avg_punctum: float,
                     discard_size: int = 12,
                     subimage_width_factor: float = 0.8,
                     notehead_h: float = 0.0) -> Optional[ReferencePoint]:
    """The pitch reference as a page-pixel point, or None if it couldn't be
    measured (glyph too small, crop off-image, or no ink in the crop).

    x is the center of the crop's own x-range, not the glyph bbox's center:
    the centroid y was computed from the rows of *that* column band, so that
    is the x it belongs to, and it's the x a curved staff line should be
    sampled at. (Rodan instead reads its staff position at the bbox's left
    edge, which is the band's left edge rather than its middle.)

    Returning None rather than a silent 0.0 is the difference from
    reference_row: a caller placing a notehead needs "no measurement" to be
    distinguishable from "measured at the bbox's top edge", so it can fall
    back to geometry instead of anchoring a whole neume on the wrong row.

    extended_rules is on unconditionally: this entry point exists for the
    decomposition path, which is the caller those extra crop rules are for.
    """
    region = reference_region(image, glyph, avg_punctum, discard_size,
                              subimage_width_factor, extended_rules=True,
                              notehead_h=notehead_h)
    if region is None:
        return None
    crop = crop_and_binarize(image, region.ulx, region.uly, region.ncols, region.nrows)
    if crop.size == 0 or not (crop != 0).any():
        return None
    return ReferencePoint(x=region.ulx + region.ncols / 2,
                          y=region.uly + row_projection_centroid(crop),
                          region=region.region)

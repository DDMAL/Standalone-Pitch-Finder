"""
Rodan-style pitch finder: reimplements the DESIGN of Rodan's heuristic
pitch-finding job --
https://github.com/DDMAL/Rodan/blob/master/rodan-main/code/rodan/jobs/heuristic_pitch_finding/PitchFinding.py
-- against Mothra's own input formats (ic_io.Glyph, staff_io.Stave, our own
manuscript image), as a second, independent algorithm to place alongside
pitch_finder.py. This is NOT a comparison against Rodan's own output --
it's a from-scratch run of an equivalent algorithm on our own data.

Kept faithful to the original design:
  - one pitch per glyph -- no multi-note decomposition, unlike pitch_finder.py
  - the glyph's pitch reference point comes from real pixel ink (a row-
    projection centroid via glyph_pixels.py), with the same per-class
    special-case cropping (virga/podatus/F-clef) Rodan uses to exclude
    stems and non-reference ink
  - three-tier stave assignment: bbox intersection -> y-bound margin
    fallback (nearest by x-distance) -> optional forced-nearest (off by
    default, matching Rodan's always_find_staff_no=False)
  - clefs snap to the nearest LINE only (never a space); notes are
    discretized line-vs-space with a "middle half of the gap counts as
    the space" rule (Rodan's space_proportion=0.5)
  - a single "current clef" propagates across the WHOLE PAGE in
    (stave, x) reading order and is silently carried over if a stave has
    no clef of its own -- deliberately NOT isolated per-stave, unlike
    pitch_finder.py's missing_clef behavior. This is one of the concrete
    design differences we wanted preserved for side-by-side comparison.

Adapted from the literal source (see the pitch-finding plan doc for the
full rationale):
  - line/space position linearly EXTRAPOLATES past the topmost/bottommost
    real detected line instead of clamping to it. Rodan's version relied
    on Miyao supplying synthetic ledger lines beyond the real 4; our own
    staff-finding doesn't synthesize those, so a literal port would just
    flatten every out-of-staff note onto the edge line. We reuse
    staff_io.Stave.continuous_step_at_y for this -- see _stave_position
    below for why rounding it is mathematically equivalent to Rodan's
    space_proportion=0.5 threshold for in-bounds positions, which means
    it gives us faithful in-bounds behavior AND correct extrapolation for
    free, without a separate hand-rolled interpolator.
  - the clef-to-line-number arithmetic doesn't hardcode a 4-line staff
    (Rodan's `6 - floor(strt_pos / 2)`); we work directly in
    staff_io's own step units instead, which are already line-count-agnostic.
  - "pitchless" glyph categories are matched against our real vocabulary
    (divisio.*, accidental.*) instead of the original's stale 'division' string.
  - every glyph gets an output record with a `reason` when no pitch could
    be found (missing_staff / missing_clef / pitchless_symbol), instead of
    being silently dropped from the output the way Rodan's does. A note
    before ANY clef has appeared anywhere on the page also gets
    missing_clef here, rather than silently defaulting to Rodan's
    hardcoded ('c', 7) bootstrap value (which is just an arbitrary
    initializer, not a documented musical decision).
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

import clef_rules
from ic_io import Glyph
from staff_io import Stave
from glyph_pixels import average_punctum, reference_row

# Tunables, mirroring Rodan's PitchFinder.__init__ defaults.
DISCARD_SIZE = 12
GET_STAFF_MARGIN = 2.0            # x avg_punctum, for the y-bound stave fallback
SUBIMAGE_WIDTH_FACTOR = 0.8
ALWAYS_FIND_STAFF_NO = False        # Rodan default: don't force an assignment

_PITCHLESS_PREFIXES = ("divisio.", "accidental.")
_CLEF_LETTER_RE = re.compile(r"^[A-Za-z]")


@dataclass
class RodanNoteResult:
    glyph_index: int
    ic: dict
    stave_id: Optional[int] = None
    stave_step: Optional[float] = None   # plays the role of Rodan's strt_pos; see _stave_position
    # The page-pixel notehead center this glyph's pitch was read from: the
    # row-projection centroid of the per-class crop (see glyph_pixels), at
    # Rodan's own reference x (the bbox's left edge). Recorded so the debug
    # overlay can mark it -- unlike stave_step it is NOT discretized, so
    # comparing the two shows how far the line/space snap had to move.
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    pitch: Optional[dict] = None
    clef_used: Optional[str] = None
    reason: Optional[str] = None
    flags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _ic_dict(g: Glyph) -> dict:
    return {
        "ulx": g.ulx, "uly": g.uly, "nrows": g.nrows, "ncols": g.ncols,
        "class_name": g.class_name, "confidence": g.confidence, "state": g.state,
    }


def _gtype(class_name: str) -> str:
    return class_name.split(".")[0]


def _clef_letter(class_name: str) -> Optional[str]:
    parts = class_name.split(".")
    if len(parts) < 2:
        return None
    m = _CLEF_LETTER_RE.match(parts[1])
    return m.group().upper() if m else None


def _is_non_music(g: Glyph) -> bool:
    return g.state == "UNCLASSIFIED" or g.class_name == "text" or g.class_name.startswith("skip.")


def _stave_bbox(stave: Stave) -> tuple:
    minx = min(ln.x_start for ln in stave.lines)
    maxx = max(ln.x_end for ln in stave.lines)
    miny = min(min(ln.y_values) for ln in stave.lines)
    maxy = max(max(ln.y_values) for ln in stave.lines)
    return minx, miny, maxx, maxy


def _intersect_area(box1: tuple, box2: tuple) -> float:
    l, b = max(box1[0], box2[0]), max(box1[1], box2[1])
    r, t = min(box1[2], box2[2]), min(box1[3], box2[3])
    if l >= r or b >= t:
        return 0.0
    return (r - l) * (t - b)


def assign_stave(glyph: Glyph, staves: list[Stave], avg_punctum: float) -> tuple[Optional[Stave], list[str]]:
    """Three-tier stave assignment, faithful to Rodan's _get_staff_no:
    1. bbox intersection (largest overlap area wins)
    2. y-bound margin fallback: any stave whose y-range (padded by
       get_staff_margin * avg_punctum) contains the glyph, closest by
       x-distance to that stave's bbox left/right edge
    3. optional forced-nearest (off by default, matching always_find_staff_no)
    """
    glyph_box = (glyph.ulx, glyph.uly, glyph.ulx + glyph.ncols, glyph.uly + glyph.nrows)

    best_area, best_stave = 0.0, None
    y_bound_candidates = []
    margin = GET_STAFF_MARGIN * avg_punctum

    for stave in staves:
        sbox = _stave_bbox(stave)
        area = _intersect_area(glyph_box, sbox)
        if area > best_area:
            best_area, best_stave = area, stave

        if not (glyph_box[1] > sbox[3] + margin or glyph_box[3] < sbox[1] - margin):
            y_bound_candidates.append((stave, sbox))

    if best_stave is not None:
        return best_stave, []

    if y_bound_candidates:
        # Matches _find_closest_y_staff_no: com_point x = offset_x + ncols (right edge).
        right_x = glyph.ulx + glyph.ncols
        stave, _ = min(y_bound_candidates, key=lambda t: min(abs(right_x - t[1][0]), abs(right_x - t[1][2])))
        return stave, ["y_bound_fallback"]

    if ALWAYS_FIND_STAFF_NO and staves:
        def rect_distance(stave):
            sbox = _stave_bbox(stave)
            dx = max(sbox[0] - glyph.ulx, 0, glyph.ulx - sbox[2])
            dy = max(sbox[1] - glyph.uly, 0, glyph.uly - sbox[3])
            return (dx ** 2 + dy ** 2) ** 0.5
        return min(staves, key=rect_distance), ["forced_nearest"]

    return None, ["missing_staff"]


def _stave_position(ref_x: float, ref_y: float, stave: Stave, is_clef: bool) -> tuple[Optional[float], list[str]]:
    """Line/space position at (ref_x, ref_y), in staff_io's own step units
    (line=even, space=odd, higher step=higher pitch -- opposite sign/anchor
    from Rodan's own page-oriented strt_pos, but the same underlying concept).

    Rounding a continuous step value to the nearest integer is mathematically
    equivalent to Rodan's space_proportion=0.5 rule: within one gap, a linear
    step value crosses each half-gap boundary at exactly the 25%/75% marks,
    which is precisely "the middle 50% of the gap counts as the space, else
    nearest line". Reusing continuous_step_at_y for this also gives correct
    linear extrapolation past the topmost/bottommost real line for free
    (seethe module docstring's "Adapted" section for why that matters here).

    Clefs additionally snap to the nearest LINE (even step), never a space.

    ref_x is clamped to the stave's own line-covered x-range before lookup:
    a glyph (very often a clef, being the leftmost symbol on its system) can
    sit to the left/right of where the detected lines actually start/end.
    This mirrors Rodan's own _gen_line_func, which falls back to a flat
    (constant-y) line whenever ref_x falls outside a line's own point range
    -- clamping x and reading the nearest line's y is the same idea.
    """
    min_x = min(ln.x_start for ln in stave.lines)
    max_x = max(ln.x_end for ln in stave.lines)
    clamped_x = min(max(ref_x, min_x), max_x)

    step, flags = stave.continuous_step_at_y(clamped_x, ref_y)
    if clamped_x != ref_x:
        flags = flags + ["x_clamped_to_stave_range"]
    if step is None:
        return None, flags
    if is_clef:
        return round(step / 2) * 2, flags
    return round(step), flags


def find_pitches_rodan(glyphs: list[Glyph], staves: list[Stave], image: np.ndarray,
                        discard_size: int = DISCARD_SIZE) -> list[RodanNoteResult]:
    music_glyphs = [g for g in glyphs if not _is_non_music(g)]
    avg_punctum = average_punctum(music_glyphs)

    results: dict[int, RodanNoteResult] = {}
    # (glyph, stave, stave_step, flags, center_x, center_y) for glyphs
    # eligible for a pitch -- collected here so they can be sorted into global
    # reading order before the clef-propagation pass, matching Rodan's
    # _sort_glyphs.
    pending: list[tuple] = []

    for g in glyphs:
        if _is_non_music(g):
            results[g.index] = RodanNoteResult(g.index, _ic_dict(g), reason="pitchless_symbol")
            continue

        stave, stave_flags = assign_stave(g, staves, avg_punctum)
        if stave is None:
            results[g.index] = RodanNoteResult(g.index, _ic_dict(g), reason="missing_staff", flags=stave_flags)
            continue

        if g.class_name.startswith(_PITCHLESS_PREFIXES):
            results[g.index] = RodanNoteResult(g.index, _ic_dict(g), stave_id=stave.stave_id,
                                                reason="pitchless_symbol", flags=stave_flags)
            continue

        gtype = _gtype(g.class_name)
        if gtype not in ("neume", "custos", "clef"):
            results[g.index] = RodanNoteResult(g.index, _ic_dict(g), stave_id=stave.stave_id,
                                                reason="pitchless_symbol", flags=stave_flags)
            continue

        ref_row = reference_row(image, g, avg_punctum, discard_size, SUBIMAGE_WIDTH_FACTOR)
        ref_x, ref_y = g.ulx, g.uly + ref_row
        step, pos_flags = _stave_position(ref_x, ref_y, stave, is_clef=(gtype == "clef"))
        flags = stave_flags + pos_flags

        if step is None:
            # Should be rare now that _stave_position clamps x into range --
            # only happens if the assigned stave has no lines at all.
            results[g.index] = RodanNoteResult(g.index, _ic_dict(g), stave_id=stave.stave_id,
                                                center_x=ref_x, center_y=ref_y,
                                                reason="no_line_coverage", flags=flags)
            continue

        pending.append((g, stave, step, flags, ref_x, ref_y))

    # Global clef propagation in (stave, x) reading order.
    pending.sort(key=lambda t: (t[1].stave_id, t[0].ulx))
    current_clef = None  # (pname, own_step); None until a real clef is seen anywhere on the page

    for g, stave, step, flags, ref_x, ref_y in pending:
        gtype = _gtype(g.class_name)

        if gtype == "clef":
            pname = _clef_letter(g.class_name)
            octave, octave_flags = clef_rules.clef_octave_for(pname)
            current_clef = (pname, step)
            results[g.index] = RodanNoteResult(
                g.index, _ic_dict(g), stave_id=stave.stave_id, stave_step=step,
                center_x=ref_x, center_y=ref_y,
                pitch={"pname": pname, "oct": octave}, clef_used=f"clef.{pname.lower()}",
                flags=flags + octave_flags,
            )
            continue

        if current_clef is None:
            results[g.index] = RodanNoteResult(g.index, _ic_dict(g), stave_id=stave.stave_id,
                                                stave_step=step, center_x=ref_x, center_y=ref_y,
                                                reason="missing_clef", flags=flags)
            continue

        clef_pname, clef_step = current_clef
        clef_octave, octave_flags = clef_rules.clef_octave_for(clef_pname)
        pname, octave = clef_rules.step_to_pitch(step - clef_step, clef_pname, clef_octave)
        results[g.index] = RodanNoteResult(
            g.index, _ic_dict(g), stave_id=stave.stave_id, stave_step=step,
            center_x=ref_x, center_y=ref_y,
            pitch={"pname": pname, "oct": octave}, clef_used=f"clef.{clef_pname.lower()}",
            flags=flags + octave_flags,
        )

    return [results[g.index] for g in glyphs]

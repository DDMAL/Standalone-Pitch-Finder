"""
Core pitch-finding algorithm.

Given the glyphs from an IC XML (ic_io.Glyph) and the staves from a
staff-finding JSON (staff_io.Stave), for each glyph:

1. Short-circuit non-music glyphs (text bboxes, Gamera junk classes) as
   pitchless -- no stave assignment attempted.
2. Assign the glyph to the nearest stave whose line span (plus a margin)
   covers the glyph's bbox-center, by x.
3. Decompose the glyph into note components using the neume shape table
   (neume_shapes.py), which already encodes each component's diatonic
   interval offset from the neume's first note. This maps the glyph's own
   vertical extent (bbox top/bottom) onto its known interval span.
4. Resolve the nearest clef on the same stave (preferring one to the left,
   i.e. already in effect when reading left-to-right) and convert each note
   component's stave step, relative to the clef's own step, into an
   absolute pitch via clef_rules.step_to_pitch.

See the pitch-finding plan doc for the full design rationale and the list
of known limitations (clef octave register is a placeholder; stave step
numbering trusts staff-finding's within_stave_index ordering).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

import clef_rules
from ic_io import Glyph
from staff_io import Stave
from neume_shapes import NeumeShapeTable

# How many extra diatonic steps beyond a stave's detected line span still
# count as "on this stave" (notes routinely sit above/below the staff).
STAVE_MARGIN_STEPS = 2


@dataclass
class NoteComponent:
    """One note of a (possibly multi-note) neume.

    center_x / center_y are the page-pixel point this component's pitch was
    read from: center_x is the glyph's horizontal center (the x the stave was
    queried at), center_y is stave_step converted back to pixels. Recording it
    is what lets the debug overlay mark each computed notehead center, so a
    wrong pitch can be told apart into "the center landed in the wrong place"
    vs "the center is right and the step-to-pitch lookup is wrong".
    """
    interval_from_first: int
    stave_step: Optional[float]
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    pitch: Optional[dict] = None


@dataclass
class ClefRef:
    class_name: str
    glyph_index: int


@dataclass
class GlyphResult:
    glyph_index: int
    ic: dict
    stave_id: Optional[int] = None
    stave_assignment_flags: list[str] = field(default_factory=list)
    note_components: list[NoteComponent] = field(default_factory=list)
    clef_used: Optional[ClefRef] = None
    reason: Optional[str] = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "glyph_index": self.glyph_index,
            "ic": self.ic,
            "stave_id": self.stave_id,
            "stave_assignment_flags": self.stave_assignment_flags,
            "note_components": [asdict(nc) for nc in self.note_components],
            "clef_used": asdict(self.clef_used) if self.clef_used else None,
            "reason": self.reason,
            "flags": self.flags,
        }


def _glyph_ic_dict(g: Glyph) -> dict:
    return {
        "ulx": g.ulx, "uly": g.uly, "nrows": g.nrows, "ncols": g.ncols,
        "class_name": g.class_name, "confidence": g.confidence, "state": g.state,
    }


def assign_stave(glyph: Glyph, staves: list[Stave]) -> tuple[Optional[Stave], list[str]]:
    """Pick the closest stave whose (margin-padded) line span covers the
    glyph's bbox-center, by x. None + ["missing_staff"] if none qualify."""
    cx, cy = glyph.center_x, glyph.center_y
    candidates = []
    for stave in staves:
        span = stave.y_span_at_x(cx)
        if span is None:
            continue
        min_y, max_y = span
        margin_px = STAVE_MARGIN_STEPS * stave.half_gap_at_x(cx)
        if min_y - margin_px <= cy <= max_y + margin_px:
            distance = stave.nearest_line_distance(cx, cy)
            candidates.append((distance, stave))
    if not candidates:
        return None, ["missing_staff"]
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1], []


def _decompose(glyph: Glyph, stave: Stave, shapes: NeumeShapeTable) -> Optional[tuple[list[NoteComponent], list[str]]]:
    """Map glyph bbox top/bottom onto the neume's known interval span.

    Returns one NoteComponent per note (step plus its page-pixel center) and
    any flags from the step interpolation. Classes missing from the shape
    table fall back to a single-note approximation (see
    approximate_unknown_shape below); this only returns None in the (now rare)
    case where the stave has no line coverage at the glyph's x at all.
    """
    intervals = shapes.intervals_for(glyph.class_name)
    approx_flags = []
    if intervals is None:
        # Not pitchless (that was already ruled out before _decompose is
        # called) but not in the neume-shape CSV either -- rather than
        # refusing outright, fall back to treating it as a single note
        # (bbox top/bottom center, same as punctum/custos) and flag it so
        # this is never mistaken for a confident, CSV-backed decomposition.
        intervals = [0]
        approx_flags = ["approximate_unknown_shape"]

    cx = glyph.center_x
    step_top, flags_top = stave.continuous_step_at_y(cx, glyph.uly)
    step_bottom, flags_bot = stave.continuous_step_at_y(cx, glyph.lry)
    flags = sorted(set(flags_top) | set(flags_bot)) + approx_flags
    if step_top is None or step_bottom is None:
        return None

    lo, hi = min(intervals), max(intervals)
    if hi == lo:
        step0 = (step_top + step_bottom) / 2
    else:
        step0 = step_bottom + (0 - lo) * (step_top - step_bottom) / (hi - lo)

    components = [NoteComponent(iv, step0 + iv, center_x=cx,
                                center_y=stave.y_at_step(cx, step0 + iv))
                  for iv in intervals]
    return components, flags


def find_pitches(glyphs: list[Glyph], staves: list[Stave], shapes: NeumeShapeTable) -> list[GlyphResult]:
    results: dict[int, GlyphResult] = {}
    # stave_id -> list of (glyph_index, center_x, class_name, pname, octave,
    # step) for clefs successfully anchored during pass 1.
    clefs_by_stave: dict[int, list[tuple[int, float, str, str, int, float]]] = {}

    # Pass 1: stave assignment + geometric decomposition (clef-independent).
    for g in glyphs:
        ic_dict = _glyph_ic_dict(g)

        if g.state == "UNCLASSIFIED" or shapes.is_pitchless(g.class_name):
            results[g.index] = GlyphResult(g.index, ic_dict, reason="pitchless_symbol")
            continue

        stave, stave_flags = assign_stave(g, staves)
        if stave is None:
            results[g.index] = GlyphResult(g.index, ic_dict, stave_assignment_flags=stave_flags, reason="missing_staff")
            continue

        decomposition = _decompose(g, stave, shapes)
        if decomposition is None:
            # Only happens if the assigned stave has no line coverage at
            # this glyph's x at all -- unknown classes no longer land here,
            # they get the approximate_unknown_shape fallback instead.
            results[g.index] = GlyphResult(
                g.index, ic_dict, stave_id=stave.stave_id,
                stave_assignment_flags=stave_flags, reason="no_line_coverage",
            )
            continue

        note_components, decomp_flags = decomposition
        result = GlyphResult(
            g.index, ic_dict, stave_id=stave.stave_id,
            stave_assignment_flags=stave_flags, note_components=note_components,
            flags=decomp_flags, reason="_pending_clef",
        )
        results[g.index] = result

        if shapes.is_clef(g.class_name):
            pname = shapes.clef_pname(g.class_name)
            octave, octave_flags = clef_rules.clef_octave_for(pname)
            note_components[0].pitch = {"pname": pname, "oct": octave}
            result.clef_used = ClefRef(g.class_name, g.index)
            result.reason = None
            result.flags.extend(octave_flags)
            clefs_by_stave.setdefault(stave.stave_id, []).append(
                (g.index, g.center_x, g.class_name, pname, octave,
                 note_components[0].stave_step)
            )

    # Pass 2: resolve clef for every non-clef glyph that has a stave + decomposition.
    for g in glyphs:
        result = results[g.index]
        if result.reason != "_pending_clef":
            continue

        clefs = clefs_by_stave.get(result.stave_id, [])
        if not clefs:
            result.reason = "missing_clef"
            continue

        left_clefs = [c for c in clefs if c[1] <= g.center_x]
        if left_clefs:
            chosen = max(left_clefs, key=lambda c: c[1])
        else:
            chosen = min(clefs, key=lambda c: abs(c[1] - g.center_x))
            result.flags.append("clef_after_glyph")

        clef_idx, _clef_x, clef_class, clef_pname, clef_octave, clef_step = chosen
        for nc in result.note_components:
            step_delta = nc.stave_step - clef_step
            pname, octave = clef_rules.step_to_pitch(step_delta, clef_pname, clef_octave)
            nc.pitch = {"pname": pname, "oct": octave}

        result.clef_used = ClefRef(clef_class, clef_idx)
        result.reason = None

    return [results[g.index] for g in glyphs]

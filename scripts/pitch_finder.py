"""
Core pitch-finding algorithm.

Given the glyphs from an IC XML (ic_io.Glyph) and the staves from a
staff-finding JSON (staff_io.Stave), for each glyph:

1. Short-circuit non-music glyphs (text bboxes, Gamera junk classes) as
   pitchless -- no stave assignment attempted.
2. Assign the glyph to the nearest stave whose line span (plus a margin)
   covers the glyph's bbox-center, by x.
3. Find ONE anchor point on the glyph and read its stave step, then place
   every other note of the neume relative to it using the neume shape table
   (neume_shapes.py), which encodes each component's diatonic interval
   offset from the neume's first note.
4. Resolve the nearest clef on the same stave (preferring one to the left,
   i.e. already in effect when reading left-to-right) and convert each note
   component's stave step, relative to the clef's own step, into an
   absolute pitch via clef_rules.step_to_pitch.

Step 3 has two anchoring modes (see _decompose):

  - bbox_span (no image): the glyph's bbox top and bottom edges are mapped
    onto the known interval span. Pure geometry, no pixel access, but it
    assumes the outermost noteheads' centers sit exactly on the ink extremes
    -- which stems, tails and ligature strokes routinely break.
  - pixel_centroid (image supplied): the anchor is the ink centroid of a
    per-class sub-region of the bbox, borrowed from rodan_pitch_finder's
    notehead finding (glyph_pixels.reference_point) -- the crop rules there
    exist precisely to exclude a virga's stem or a podatus's second head, plus
    the torculus rule this module needs and Rodan has no use for (see
    glyph_pixels' extended_rules). Rodan uses that point as its one pitch per glyph;
    here it instead anchors this module's own stave/clef/decomposition math,
    which is the combination of the two algorithms' better halves.

See the pitch-finding plan doc for the full design rationale and the list
of known limitations (clef octave register is a placeholder; stave step
numbering trusts staff-finding's within_stave_index ordering).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

import clef_rules
from ic_io import Glyph
from staff_io import Stave
from neume_shapes import NeumeShapeTable, SOURCE_CLASS_NAME
from glyph_pixels import (average_punctum, notehead_height, reference_point,
                          ReferencePoint, REGION_TOP, REGION_BOTTOM_LEFT)

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
class Anchor:
    """The single point a glyph's whole decomposition was measured from.

    Recorded per glyph because every note of the neume is placed relative to
    it: if a neume's pitches are uniformly a step off, the anchor is where to
    look, and `interval` says which note of the neume the point was taken to
    be (so a wrong anchor *role* can be told from a wrong anchor *position*).
    """
    source: str                   # "pixel_centroid" | "bbox_span"
    region: Optional[str]         # glyph_pixels REGION_* for pixel_centroid, else None
    x: float
    y: float
    stave_step: float
    interval: float               # the neume interval this point represents


@dataclass
class GlyphResult:
    glyph_index: int
    ic: dict
    stave_id: Optional[int] = None
    stave_assignment_flags: list[str] = field(default_factory=list)
    note_components: list[NoteComponent] = field(default_factory=list)
    anchor: Optional[Anchor] = None
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
            "anchor": asdict(self.anchor) if self.anchor else None,
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


def _anchor_interval(region: str, intervals: list[int]) -> float:
    """Which note of the neume a pixel reference point stands for, given the
    crop region it was measured from.

    This is the join between the two algorithms: Rodan's crop rules already
    encode which notehead they isolate, but Rodan never has to name it (one
    pitch per glyph). Decomposing a neume does, because every other note is
    placed relative to this one.

      - bottom-left crop (the ascending ligatures -- podatus*, pescephalicus*,
        scandicus*, torculus*): the ligature is
        drawn starting from its bottom-left head, so that crop lands on the
        neume's FIRST note -- interval 0 by definition. For the ascending
        ligatures the first note is also the lowest, so this reads the same as
        min(intervals); for a torculus it does not (torculus23 = [0, 1, -1]
        ends below where it started), and binding to the first note is what
        stays correct for both.
      - top crop: the HIGHEST note -- the notehead above a virga's stem, and
        the top-left head of a clivis, which descends and so has its first
        note and its highest note in the same place.
      - full bbox: an ink centroid over the whole shape, which belongs to no
        single note. Best available reading is the middle of the note span,
        so a fractional interval is returned. Single-note classes collapse to
        0 here, and `neume.oblique*` -- the one multi-note class left without a
        crop rule -- gets the honest midpoint rather than a pretend notehead,
        which for a symmetric diagonal parallelogram is what its centroid
        actually is.
    """
    lo, hi = float(min(intervals)), float(max(intervals))
    if region == REGION_BOTTOM_LEFT:
        return 0.0
    if region == REGION_TOP:
        return hi
    return (lo + hi) / 2


def _decompose(glyph: Glyph, stave: Stave, shapes: NeumeShapeTable,
               ref: Optional[ReferencePoint] = None
               ) -> Optional[tuple[list[NoteComponent], list[str], Anchor]]:
    """Place every note of the neume from one anchor plus its interval list.

    With ref, the anchor is that measured ink centroid (pixel_centroid mode);
    without it, the anchor is interpolated from the bbox's top/bottom edges
    against the interval span (bbox_span mode). Either way the interval list
    supplies the exact integer offsets, so one anchor determines all notes.

    Returns (one NoteComponent per note, flags, the anchor used). Classes
    missing from the shape table fall back to a single-note approximation
    (see approximate_unknown_shape below); this only returns None in the (now
    rare) case where the stave has no line coverage at the glyph's x at all.
    """
    intervals, source = shapes.intervals_with_source(glyph.class_name)
    approx_flags = []
    if intervals is None:
        # Neither in the neume-shape CSV nor decodable from the class name
        # (pitchless classes were already ruled out before _decompose is
        # called) -- rather than refusing outright, fall back to treating it
        # as a single note (same as punctum/custos) and flag it so this is
        # never mistaken for a confident, CSV-backed decomposition.
        intervals = [0]
        approx_flags = ["approximate_unknown_shape"]
    elif source == SOURCE_CLASS_NAME:
        # Decomposed, but off the class name rather than the CSV's MEI. Worth
        # saying out loud: neume.distropha's unison is still an open question
        # in the plan doc, and this is the flag that finds every glyph whose
        # answer depends on it.
        approx_flags = ["shape_from_class_name"]

    if ref is not None:
        placement = _anchor_from_pixels(glyph, stave, intervals, ref)
    else:
        placement = _anchor_from_bbox_span(glyph, stave, intervals)
    if placement is None:
        return None
    anchor, flags = placement

    components = []
    for iv in intervals:
        step = anchor.stave_step + (iv - anchor.interval)
        components.append(NoteComponent(iv, step, center_x=anchor.x,
                                        center_y=stave.y_at_step(anchor.x, step)))
    return components, flags + approx_flags, anchor


def _anchor_from_pixels(glyph: Glyph, stave: Stave, intervals: list[int],
                        ref: ReferencePoint) -> Optional[tuple[Anchor, list[str]]]:
    """Anchor on a measured ink centroid (rodan-style notehead finding)."""
    anchor_x = ref.x
    step, flags = stave.continuous_step_at_y(anchor_x, ref.y)
    if step is None:
        # Stave assignment guaranteed line coverage at the bbox center, but
        # the crop's own x-band can still fall off the end of the detected
        # lines (a clef sitting left of where line-fitting starts). Read the
        # measured row at the center's x rather than discarding the glyph --
        # the lines are near-horizontal, so the row is the part that matters.
        anchor_x = glyph.center_x
        step, flags = stave.continuous_step_at_y(anchor_x, ref.y)
        if step is None:
            return None
        flags = flags + ["anchor_x_fell_back_to_center"]

    anchor = Anchor(source="pixel_centroid", region=ref.region, x=anchor_x, y=ref.y,
                    stave_step=step, interval=_anchor_interval(ref.region, intervals))
    return anchor, flags


def _anchor_from_bbox_span(glyph: Glyph, stave: Stave,
                           intervals: list[int]) -> Optional[tuple[Anchor, list[str]]]:
    """Anchor by mapping the bbox's top/bottom edges onto the interval span.

    The geometry-only path: no image needed, but it takes the ink extremes for
    notehead centers, so any stem or tail that overshoots the outermost head
    biases every note of the glyph.
    """
    cx = glyph.center_x
    step_top, flags_top = stave.continuous_step_at_y(cx, glyph.uly)
    step_bottom, flags_bot = stave.continuous_step_at_y(cx, glyph.lry)
    flags = sorted(set(flags_top) | set(flags_bot))
    if step_top is None or step_bottom is None:
        return None

    lo, hi = min(intervals), max(intervals)
    if hi == lo:
        step0 = (step_top + step_bottom) / 2
    else:
        step0 = step_bottom + (0 - lo) * (step_top - step_bottom) / (hi - lo)

    anchor = Anchor(source="bbox_span", region=None, x=cx,
                    y=stave.y_at_step(cx, step0), stave_step=step0, interval=0.0)
    return anchor, flags


def find_pitches(glyphs: list[Glyph], staves: list[Stave], shapes: NeumeShapeTable,
                 image: Optional[np.ndarray] = None) -> list[GlyphResult]:
    """Pitch every glyph on the page.

    image is the page's own pixels. Supplying it switches note anchoring from
    bbox geometry to rodan-style per-class ink centroids (see the module
    docstring); everything downstream -- stave assignment, step lookup,
    per-stave clef resolution, interval decomposition -- is unchanged either
    way. Omit it and the module runs on geometry alone, exactly as before.
    """
    results: dict[int, GlyphResult] = {}
    # stave_id -> list of (glyph_index, center_x, class_name, pname, octave,
    # step) for clefs successfully anchored during pass 1.
    clefs_by_stave: dict[int, list[tuple[int, float, str, str, int, float]]] = {}

    # Rodan's proxy for one notehead's width, which sizes every per-class crop.
    # Measured over the whole page once. A page with no punctum or virga at all
    # gives 0, which would collapse the crops to 1px -- fall back to geometry
    # rather than anchor on a sliver of ink.
    avg_punctum = average_punctum(glyphs) if image is not None else 0.0
    use_pixel_anchor = image is not None and avg_punctum > 0
    # How deep one notehead is on this page, which is what sizes the
    # first-head crop. Falls back to avg_punctum (a *width*) inside
    # reference_region when the page has no punctum or inclinatum to measure.
    notehead_h = notehead_height(glyphs) if use_pixel_anchor else 0.0

    # Pass 1: stave assignment + anchoring/decomposition (clef-independent).
    for g in glyphs:
        ic_dict = _glyph_ic_dict(g)

        if g.state == "UNCLASSIFIED" or shapes.is_pitchless(g.class_name):
            results[g.index] = GlyphResult(g.index, ic_dict, reason="pitchless_symbol")
            continue

        stave, stave_flags = assign_stave(g, staves)
        if stave is None:
            results[g.index] = GlyphResult(g.index, ic_dict, stave_assignment_flags=stave_flags, reason="missing_staff")
            continue

        ref, ref_flags = None, []
        if image is not None:
            if use_pixel_anchor:
                ref = reference_point(image, g, avg_punctum, notehead_h=notehead_h)
            if ref is None:
                # Too small to analyze, off-image, or no ink in the crop.
                # Geometry still works; say so rather than anchoring on a
                # centroid that was never measured.
                ref_flags = ["pixel_anchor_unavailable"]

        decomposition = _decompose(g, stave, shapes, ref)
        if decomposition is None:
            # Only happens if the assigned stave has no line coverage at
            # this glyph's x at all -- unknown classes no longer land here,
            # they get the approximate_unknown_shape fallback instead.
            results[g.index] = GlyphResult(
                g.index, ic_dict, stave_id=stave.stave_id,
                stave_assignment_flags=stave_flags, flags=ref_flags,
                reason="no_line_coverage",
            )
            continue

        note_components, decomp_flags, anchor = decomposition
        result = GlyphResult(
            g.index, ic_dict, stave_id=stave.stave_id,
            stave_assignment_flags=stave_flags, note_components=note_components,
            anchor=anchor, flags=ref_flags + decomp_flags, reason="_pending_clef",
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

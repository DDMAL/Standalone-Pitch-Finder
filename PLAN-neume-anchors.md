# Plan: per-class note anchoring for `pitch_finder.py`

Status: proposed, not implemented. Supersedes the bbox-span decomposition in
[`scripts/pitch_finder.py`](scripts/pitch_finder.py) `_decompose`.

## Problem

`_decompose` maps the glyph bbox's top and bottom edges linearly onto the
neume's interval span. That assumes the two extreme noteheads' *centers* sit
exactly on the bbox's ink extremes. Neither half holds:

- **Stems and tails overshoot.** A virga's stem, a clivis's left stem, and a
  podatus's descending tail all extend past the outermost notehead, so the
  bbox edge on that side is not a note position at all.
- **Ink extreme ≠ head center.** Even on a clean side, the bbox edge is the
  *top of the ink*, about half a notehead above the head's center.

Both errors are large relative to the quantity being measured. On
`McGill_MS234-064` the median `neume.punctum` bbox is 25 px tall against
20.9 px per diatonic step, so **one notehead is ~1.2 steps** and half a
notehead is 0.6 steps — past the 0.5-step rounding threshold in
`clef_rules.step_to_pitch`. A half-notehead error is a whole wrong pitch.

### Measured evidence

Median step disagreement against `rodan_pitch_finder.py` (which crops per
class) on `McGill_MS234-064`, and the vertical ink distribution of each class
(bbox split into 10 bands, top→bottom, share of ink):

| class | n | ours − rodan | peak band | profile |
|---|---|---|---|---|
| `neume.punctum` | 39 | +0.03 | 5 | symmetric — agrees |
| `neume.distropha` | 8 | +0.05 | 5 | symmetric — agrees |
| `neume.liquescent.up` | 4 | +0.04 | 4 | agrees |
| `neume.virga` | 56 | **−0.70** | 2 | head at band 2, stem trails ~9%/band to band 8 |
| `neume.clivis2` | 20 | **+0.88** | 3 | mass toward top, tail below |
| `neume.podatus2b` | 12 | **−1.10** | 2 and 5 | **bimodal**, ~25% of ink *below* the lower head |
| `neume.podatus3` | 1 | **−1.17** | 7 (and 1–2) | heads at ~15% and ~75%, bbox extends below |
| `custos` | 4 | +0.13 | flat | 140 px = 6.7 steps tall; both agree *and* both are wrong |

The classes that agree are exactly the single-head symmetric ones. Every
class with a stem, tail, or ligature disagrees, and the sign of the
disagreement matches the direction of the overshoot. `custos` is the warning
case: two methods that both average the whole shape agree with each other
while both being wrong.

## Prerequisite bug: `@intm` is relative to the *previous* note

[`neume_shapes.py:36`](scripts/neume_shapes.py#L36) reads each `intm` as an
offset from the neume's first note. MEI defines `@intm` on `<nc>` as the
melodic interval from the **preceding** component. `_extract_intervals` must
accumulate.

This must be fixed first, because the interval list is the input to every
anchor rule below, and it also changes *which component is the vertical
extreme* (`torculus33` goes from span [−2,+2] to [0,+2]).

| class | MEI | current | correct |
|---|---|---|---|
| `neume.torculus22` | `1S, -1S` | `[0, 1, -1]` | `[0, 1, 0]` |
| `neume.torculus23` | `1S, -2S` | `[0, 1, -2]` | `[0, 1, -1]` |
| `neume.torculus24` | `1S, -3S` | `[0, 1, -3]` | `[0, 1, -2]` |
| `neume.torculus32` | `2S, -1S` | `[0, 2, -1]` | `[0, 2, 1]` |
| `neume.torculus33` | `2S, -2S` | `[0, 2, -2]` | `[0, 2, 0]` |
| `neume.torculus34` | `2S, -3S` | `[0, 2, -3]` | `[0, 2, -1]` |
| `neume.torculus42` | `3S, -1S` | `[0, 3, -1]` | `[0, 3, 2]` |
| `neume.torculus43` | `3S, -2S` | `[0, 3, -2]` | `[0, 3, 1]` |
| `neume.scandicus22a/b/c` | `1S, 1S` | `[0, 1, 1]` | `[0, 1, 2]` |
| `neume.scandicus23` | `1S, 2S` | `[0, 1, 2]` | `[0, 1, 3]` |
| `neume.scandicus33` | `2S, 2S` | `[0, 2, 2]` | `[0, 2, 4]` |

Three independent confirmations:

1. **Cheatsheet images.** In `009r scandicus23.png` the three heads sit in a
   space, on the next line up, then on the line above that — +1 then +2 more,
   i.e. +3 from the first. `009r torculus22.png` and `114r torculus33.png`
   show notes 1 and 3 at the *same* height, which only the cumulative
   reading produces.
2. **Class names decode.** `torculusAB` = up an *A*th then down a *B*th
   relative to the preceding note; `scandicusAB` = up an *A*th then up a
   *B*th. Under the current reading `scandicus` cannot ascend — it emits a
   repeated pitch.
3. **A generator reproduces the whole CSV.** Deriving intervals from the
   digits in the class name (`podatusN`→`[0, N-1]`, `clivisN`/`obliqueN`→
   `[0, -(N-1)]`, `torculusAB`→`[0, A-1, A-B]`, `scandicusAB`→
   `[0, A-1, A+B-2]`) reproduces all 33 multi-note classes in the CSV
   exactly, but only under the cumulative reading.

That generator is worth keeping as a cross-check (see Tests) and as coverage
for classes the CSV omits.

## Design

Stop using the bbox span. The interval list is already exact integers and the
stave already yields exact px-per-step, so **one anchor determines every
note**. Consulting the second edge only lets stems corrupt the result.

```
anchor_y     = anchor_from(glyph, spec, metrics)      # per-class rule
anchor_step  = stave.continuous_step_at_y(cx, anchor_y)
step_i       = anchor_step + (interval_i - interval_at_anchor)
```

The spec splits into a **structural** part (per class, stable across scribal
styles) and a **metric** part (per page, measured, never hardcoded). That
split is what makes it portable: comparing `McGill_MS234-064` with
`GentAnt1475_0017_AC_rightcrop`, the anchor *side* is stable across both
while the *magnitude* is not — the Gent page draws the same classes much
smaller and its `clivis2` has no long descending stem at all.

### Structural: the anchor table

```python
@dataclass(frozen=True)
class AnchorSpec:
    edge: str             # "top" | "bottom" | "center"
    x_region: str         # "full" | "left" | "right"
    inset_noteheads: float  # head center's distance from that edge, in noteheads
    snap: str = "none"    # "none" | "line"
```

The anchored component is *derived*, not indexed: a `bottom` anchor binds the
min-interval component, a `top` anchor the max-interval one. After the
`@intm` fix this is always correct without a per-class index.

| class pattern | edge | x_region | inset | evidence |
|---|---|---|---|---|
| `neume.punctum`, `neume.inclinatum` | center | full | 0 | symmetric profile, peak band 5 |
| `neume.distropha` | center | full | 0 | symmetric; unison by definition |
| `neume.virga`, `neume.reversevirga` | top | full | 0.5 | peak band 2, stem trails to band 8 |
| `neume.clivis*` | top | full | 0.5 | `009r clivis3/4`: left stem descends past the lower head |
| `neume.oblique*` | top | left | 0.5 | diagonal ligature; higher note is top-left |
| `neume.podatus*`, `neume.pescephalicus*` | bottom | left | 0.5 | bimodal profile; matches Rodan's bottom-left crop |
| `neume.scandicus*` | bottom | left | 0.5 | `114r scandicus32`: bottom-left square is the lowest note |
| `neume.torculus*` | bottom | full | 0.5 | `009r torculus22/33`: two clean bottom heads, thin top stroke |
| `neume.liquescent.up/down` | top | full | 0.5 | `liquescent.down` peak band 2 |
| `clef.*` | center | full | 0 | + `snap="line"` (see below) |
| `custos` | — | — | — | **unresolved**, see Open questions |

### Metric: per-page measurements

New `page_metrics.py`:

```python
@dataclass
class PageMetrics:
    notehead_h: float   # median bbox height of neume.punctum / neume.inclinatum
    notehead_w: float
```

Fallback when a page has no punctum or inclinatum: the staff JSON's
`scale_unit`. On `McGill_MS234-064` `scale_unit` is 25.0 and the measured
median punctum height is 25.0 — but that is one page, so the fallback needs
confirming on more before it is trusted (see Open questions).

`inset_noteheads` is multiplied by `notehead_h` to get pixels, so the table
transfers between manuscripts of different scale unchanged.

### Pixel refinement (optional, hybrid)

`find_pitches(glyphs, staves, shapes, image=None, metrics=None)` — the module
stays runnable with no image, and gets more accurate with one.

When `image` is supplied, crop the anchor sub-region (the `edge`/`x_region`
band, ~1.2 × `notehead_h` deep) and take the ink centroid via the existing
`glyph_pixels.crop_and_binarize` + `row_projection_centroid`, using that as
`anchor_y` in place of `bbox_edge ± inset`.

Bound the refinement: if the refined anchor moves more than one notehead from
the geometric one, distrust it, keep the geometric value, and flag
`anchor_refine_rejected`. Otherwise flag `anchor_pixel_refined`. This guards
against ink bleed and a neighbouring glyph's ink inside the crop.

### Clefs

Clefs currently take the bbox center with no line snapping, so a clef error
shifts every pitch on its stave. Add `snap="line"`: round the clef's anchor
step to the nearest **even** step, as `rodan_pitch_finder._stave_position`
already does. Flag `clef_snap_large` when the snap moves more than 0.5 step —
that indicates a bad clef bbox or a bad stave.

### New flags

- `span_mismatch` — anchor + intervals predict a bbox span; flag when it
  differs from the observed span by more than one notehead. This is the
  closest thing to a correctness check available without ground truth, and it
  is only possible *because* the second edge is no longer an input.
- `anchor_unknown_class` — no spec matched; falls back to center. Replaces
  today's `approximate_unknown_shape`.
- Retained unchanged: `missing_staff`, `missing_clef`, `sparse_stave_lines`.

## Also in scope (found while investigating)

**Pitchless detection is membership-based, so unknown non-music classes get
pitches.** `neume_shapes.load_neume_shapes` only adds `divisio.*` /
`accidental.*` to `pitchless_classes` for rows *present in the CSV*.
`divisio.maior` and `staff` appear in real IC output but not the CSV, so
`is_pitchless` returns False and they fall through to the unknown-shape
fallback and receive a pitch. Make the test prefix-based, matching
`rodan_pitch_finder._PITCHLESS_PREFIXES`.

**Classes in real IC output but absent from the CSV.** `neume.clivis1` (18 on
McGill), `neume.distropha` (8), `neume.clivis2a`/`2b`, `neume.podatus2`,
`clef.f2`, `clef.g`, `neume.scandicus32` (in the cheatsheet images but not
the CSV). Handle by normalizing a trailing letter variant (`podatus2b` →
`podatus2`) before lookup, then falling back to the name-derived generator.

**The debug viz hides multi-note output.** ~~`run_pitch_finding._render_debug_viz`
labels `note_components[0]` only, so a decomposed torculus renders as one
pitch.~~ **Done, ahead of the rest of this plan.** All components are labelled,
and `viz_utils.draw_note_center` marks each computed notehead center
(`NoteComponent.center_x/center_y`, via the new `Stave.y_at_step`), which is
what makes the anchor rule below visually auditable once it lands: the marker
already draws wherever `_decompose` puts the note, so replacing the bbox-span
decomposition with a per-class anchor needs no viz change.

## Sequencing

| Phase | Work | Verifiable by |
|---|---|---|
| 0 | `@intm` cumulative fix; prefix-based pitchless test | generator-vs-CSV test; `divisio.maior` no longer pitched |
| 1 | `page_metrics.py`; anchor table; rewrite `_decompose` (geometry only) | per-class step diff vs Rodan collapses |
| 2 | Optional pixel refinement + rejection bound | refined vs geometric agree within a notehead |
| 3 | `span_mismatch` / clef snap / new flags; debug viz shows all components | flag counts on both sample pages |
| 4 | Regenerate sample outputs, three-way compare | debug images |

Phase 0 is independent of the redesign and worth landing on its own.

## Tests

- **Generator vs CSV** — name-derived intervals equal CSV-derived intervals
  for all 33 overlapping classes. This single test would have caught the
  `@intm` bug, and guards it permanently.
- **Anchor unit tests** on synthetic staves — a virga whose bbox top sits on
  a line resolves to *that* line's step, not one step below; a clivis with a
  stem extending 3 steps below the lower head is unaffected by the stem.
- **Regression against the measured bias** — on `McGill_MS234-064` the median
  step difference vs Rodan for `virga` / `clivis2` / `podatus2b` must fall
  below ~0.35 steps, from today's −0.70 / +0.88 / −1.10.
- **No silent pitches** — `custos`, `divisio.maior`, and unknown classes are
  flagged or pitchless, never silently pitched.

## Open questions

These need a musicologist, not a heuristic:

1. **`custos` anchor.** 140 px (6.7 steps) tall with a nearly flat ink
   profile — a long stroke that dominates any centroid. Both current
   algorithms average the whole shape and agree with each other while almost
   certainly being wrong. Needs someone to state which end carries the pitch.
2. **`neume.clivis1`** — 18 instances on McGill, two equal-height stems. The
   generator gives `[0, 0]` (a unison clivis). Plausible from the ink profile
   but unconfirmed.
3. **`scandicus22a` / `22b` / `22c`** share intervals `[0, 1, 2]` but are
   drawn differently, and Rodan singles out `22b` for a bottom-left crop.
   Do they need three different anchor specs?
4. **Liquescents** — does the liquescent tail carry its own pitch, or is it
   ornamental? Affects whether `liquescent.*` is one component or two.
5. **`scale_unit` as a notehead-height fallback** — exact match on one page
   is not evidence. Confirm on more pages or drop the fallback.
6. **`CLEF_OCTAVE_REFERENCE`** (C=4/F=3/G=4) remains an unvalidated
   placeholder, unchanged by this plan.

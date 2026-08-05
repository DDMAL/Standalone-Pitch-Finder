# Plan: per-class note anchoring for `pitch_finder.py`

Status: Phase 0's `@intm` fix, the shape-table lookup fallbacks, and the
first-head anchor rule for the ascending ligatures (`neume.podatus*`,
`neume.pescephalicus*`, `neume.scandicus*`, `neume.torculus*`) and the
descending one (`neume.clivis*`) are implemented; the rest is proposed. Supersedes the bbox-span decomposition in
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

## Prerequisite bug: `@intm` is relative to the *previous* note — **fixed**

`_extract_intervals` read each `intm` as an offset from the neume's first
note. MEI defines `@intm` on `<nc>` as the melodic interval from the
**preceding** component, so it now accumulates.

This had to be fixed first, because the interval list is the input to every
anchor rule below, and it also changes *which component is the vertical
extreme* (`torculus33` goes from span [−2,+2] to [0,+2]).

It also could not land alone. The old span was symmetric for `torculusAB` with
A = B, which made `_anchor_interval`'s `REGION_FULL` midpoint come out at
exactly 0 and so accidentally reasonable. Under the correct intervals that
midpoint moves to +0.5 … +1.5, which would have shifted every note of those
neumes *down* by that much — the interval fix on its own regresses the pages
it is meant to fix. Hence the torculus anchor rule below shipped with it.

| class | MEI | was | now |
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
| `neume.clivis*` | top | left | — | **implemented**, see below |
| `neume.oblique*` | top | left | 0.5 | diagonal ligature; higher note is top-left |
| `neume.podatus*`, `neume.pescephalicus*` | bottom | left | 0.5 | **implemented**, see below |
| `neume.scandicus*` | bottom | left | 0.5 | **implemented**; `114r scandicus32`: bottom-left square is the lowest note |
| `neume.torculus*` | bottom | left | — | **implemented**, see below |
| `neume.liquescent.up/down` | top | full | 0.5 | `liquescent.down` peak band 2 |
| `clef.*` | center | full | 0 | + `snap="line"` (see below) |
| `custos` | — | — | — | **unresolved**, see Open questions |

### The first-head rule, as implemented

One rule now covers every ligature with separable heads
(`glyph_pixels._head_ink_region`): crop the left column band, split its row
profile into runs of ink, keep the runs heavy enough to be noteheads, and take
the **outermost** survivor, capped to one notehead deep. That point is the
neume's first note, so nothing has to be derived from the interval list.

Which end is "outermost" is the only per-family part:

| classes | end of the left band | binds to |
|---|---|---|
| `FIRST_HEAD_INK_CLASSES` — `podatus*`, `pescephalicus*`, `scandicus*`, `torculus*` | **bottom**: an ascending ligature is drawn from its lowest head | `REGION_BOTTOM_LEFT` → interval 0 |
| `TOP_HEAD_INK_CLASSES` — `clivis*` | **top**: a descending ligature is drawn from its highest head | `REGION_TOP` → `max(intervals)`, which for a descent *is* the first note |

It replaced two different broken positionings, which is why it is one rule:

**The bbox bottom is not the first head's bottom.** Rodan hangs the band off
the bbox's bottom edge, but that edge belongs to whatever ink descends
furthest: the ligature's right-hand stroke on a podatus, and *note 3* on a
torculus whose descent outruns its ascent (`torculus23/24/34`, e.g.
`[0, 1, -2]`). Measured on `McGill_MS234-064`'s podatus, the bbox bottom sits
a median 22 px — 0.9 noteheads, a full diatonic step — below where the first
head's ink actually ends, so the band slid past the head and kept only its
lower rows. That is the reported "podatus centroid is a bit low": 0.27 steps
low at the median, 0.96 at the worst, against a 0.5-step rounding threshold.

**Staff lines are ink too.** They cross the column band as densely as a
notehead does and drag any fixed-depth centroid towards themselves. Runs are
weighed by total ink rather than counted, which separates them by roughly an
order of magnitude (2–3 rows against 15–25) with no threshold to carry
between manuscripts.

**The clivis had the same bug in mirror, and worse.** Its first note came off
the `REGION_FULL` midpoint — the left band's whole-height centroid read as the
point *between* the two notes — so the error scaled with the neume's span:
−0.11 steps on a `clivis2a`, +0.60 on a `clivis3b`, **+1.15 on the `clivis4b`**,
which put that note *above the glyph's own bbox*. A computed notehead outside
the ink it was measured from is wrong no matter which oracle you trust, and it
is now asserted against directly
(`test_wide_clivis_keeps_its_first_note_inside_the_glyph`); across both sample
pages all 315 note centers on multi-note glyphs land inside their bbox.

The depth cap is `notehead_height()` — the median bbox height of the page's
`punctum`/`inclinatum` — not `average_punctum`, which is a *width*: on
`McGill_MS234-064` that is 35 px against a 25 px notehead, so "one notehead
deep" came out 40% too deep and reached into the stroke above the head.

The cap is a **full** notehead going up and `TOP_HEAD_DEPTH_FRACTION` = 0.75 of
one coming down, because it is doing a structurally different job in each
direction. A podatus keeps its ligature stroke on the *right*, outside the left
band, so the band's lowest run is the bare notehead — a median 0.68 noteheads
tall on both pages, never past 1.3 — and the cap almost never engages. A
clivis's left stem descends *inside* the band, fused to the head: that run
measures 1.68 noteheads on McGill, 64% of them past 1.3, so the cap is the only
thing separating head from stem and every pixel of it moves the anchor. 0.75 is
mid-plateau — 0.70–0.75 both leave 92% of clivis second notes inside half a
step against 88% at a full notehead, with a signed bias of −0.01 steps.

Verified against ink the anchor never touches, one page at a time, because
neither check is valid on both:

| page | oracle | why the other one doesn't apply | n | median \|err\| | within ½ step |
|---|---|---|---|---|---|
| `McGill_MS234-064` | note 2 off the opposite column band | its bbox bottom is a descending stroke, so bbox geometry is not a note position | 12 | 0.43 → **0.16** | 55% → **100%** |
| `Breviarium_ad_usum` | bbox bottom − ½ notehead | a notehead spans 1.87 steps there, so two heads a step apart overlap and no run split resolves them | 42 | 0.46 → **0.31** | 55% → **88%** |
| both, `clivis*` | note 2 off the opposite column band | — (valid on both once the runs are capped; see below) | 75 | 0.15 → **0.15** | 93% → 92% |

A third page settles it. `GentAnt1475_0017_AC_rightcrop` was never used to
choose any of these constants, and its two-band oracle is weak (a notehead
spans 2.4 steps at 5.4 px/step), but old-vs-new on the *same* oracle is still a
fair comparison — and there the change is not marginal at all:

| class | n | median \|err\| | within ½ step |
|---|---|---|---|
| `clivis2` | 27 | 0.68 → **0.28** | 26% → **78%** |
| `clivis3` | 2 | 0.58 → **0.27** | 50% → **100%** |
| `clivis4` | 1 | 0.65 → **0.19** | 0% → **100%** |
| `podatus2a` | 15 | 0.60 → **0.33** | 47% → **93%** |
| `podatus3`, `podatus4` | 3 | 1.79 → 1.80 | unchanged, see below |
| **total** | 48 | 0.63 → **0.30** | 33% → **81%** |

Three glyphs there are untouched and still wrong by more than a step, and they
are the same three whose second note lands *outside* the bbox — before and
after. The anchor is on the first head; what fails is the class's span against
the drawn glyph, so either the classification or the bbox is wrong and no
anchor rule can fix it. This is precisely what the proposed `span_mismatch`
flag is for, and it is still unimplemented.

The clivis row on the two tuning pages is a wash in aggregate and that is the
honest summary: the
midpoint rule is accurate wherever the neume is narrow, and `clivis2` alone is
48 of those 75 glyphs. What moves is the tail — `clivis4b` goes 1.08 → **0.08**
steps and `clivis3b` 0.49 → **0.17**, while `clivis2b` and `clivis3a` give back
about 0.08 steps between them. Trading a bounded regression on the narrow
subclasses for eliminating whole-wrong-pitch errors on the wide ones is the
point, and unlike the midpoint it is a rule rather than a coincidence.

An earlier pass recorded this comparison as unmeasurable, because the two-band
oracle returned head separations near 0 for McGill's clivis where the class
names call for 1–3 steps. That was the oracle's fault, not the page's: without
the one-notehead cap both bands' runs span the glyph's full height (McGill
draws its clivis as a Π, with a stem at each end) and the two centroids
collapse onto each other. Capped, the oracle reproduces the class intervals on
both pages — `clivis2a` 0.86, `clivis2b` 0.79, `clivis3a` 1.57, `clivis3b`
1.83, `clivis4b` 2.86 against 1, 1, 2, 2, 3 — and is sound to measure against.

Nothing else moved: 60 of 288 glyphs changed on McGill and 85 of 223 on
Breviarium, all of them podatus, clivis, torculus, or a class that gained notes
from the lookup fallbacks below. `rodan_pitch_finder`'s output is byte-identical
on both pages — the extra rules stay behind `extended_rules`, off for
`reference_row`.

### Shape-table lookup fallbacks, as implemented

`NeumeShapeTable.intervals_with_source` now tries three lookups and reports
which one hit, so the caller can flag anything below CSV-backed:

1. the CSV verbatim → `SOURCE_CSV`;
2. the CSV with a trailing variant letter dropped → `SOURCE_CSV_VARIANT`
   (`neume.clivis2a` and `neume.clivis2b` are two ways of *drawing* a
   clivis2 and sound identical, so they inherit its `[0, -1]`);
3. the class name decoded directly → `SOURCE_CLASS_NAME`, flagged
   `shape_from_class_name`, covering both the digit scheme and the
   repeated-pitch neumes (`distropha` → `[0, 0]`) that carry no digits.

Step 3 refuses any digit of 1 — an interval of a unison — because a ligature
of two noteheads on one pitch is not a neume (see open question 2).

This is what the reported "unable to deal with `neume.distropha`,
`neume.clivis**`" was: every one of those classes missed the exact-name
lookup, fell through to the single-note `approximate_unknown_shape` path, and
was emitted as **one** note placed on the glyph's whole-shape ink centroid. On
`McGill_MS234-064` that was 46 glyphs — 31 clivis variants, 7 distropha, 5
more clivis, a podatus3b — now decomposed, +46 note components on the page.

### The torculus rule, as first implemented

This row started as `bottom | full | 0.5` and changed on contact with the
subclasses. Two corrections:

**`x_region` is `left`, not `full`.** A full-width bottom band contains *two*
heads — notes 1 and 3 — and those are only at the same height when A = B
(`torculus22`, `torculus33`). For every other subclass its centroid is an
average of two different pitches, belonging to neither. The left band contains
note 1 alone, and note 1 is interval 0 by definition, so nothing has to be
derived. `_anchor_interval` binds `REGION_BOTTOM_LEFT` to the neume's **first**
note rather than to `min(intervals)`; for podatus and scandicus those are the
same component, so that is a no-op there and correct here.

**The band is positioned by the left band's own ink, not the bbox bottom.**
A bbox-bottom band assumes the first note is also the lowest. That fails
whenever the descent outruns the ascent (`torculus23/24/34`, e.g. `[0, 1, -2]`),
where the bbox bottom is set by note 3 on the far side of the glyph — the band
would sit one or two steps below note 1 and miss it completely. Cropping the
left column band, trimming to *its* ink extent, then taking the bottom
notehead of that is the same trick `_f_clef_right_region` already uses.
(`glyph_pixels._first_head_ink_region`.)

The extra rule is gated behind `extended_rules`, on for `reference_point`
(the decomposition path) and off for `reference_row`, so `rodan_pitch_finder`
keeps Rodan's own crop rules — Rodan has no torculus case, and a baseline
that silently inherits this module's opinions is not a baseline. Verified: the
Rodan output JSON is byte-identical across the change.

Measured on the two sample pages, against note 3 read independently off the
bottom-**right** band (a mirror of the rule, and ink the anchor never touches):

| page | n | before, median \|err\| | after | within ½ step, before → after |
|---|---|---|---|---|
| `Breviarium_ad_usum` | 10 | 0.60 steps | **0.10** | 1/10 → 10/10 |
| `McGill_MS234-064` | 3 | 1.41 steps | **0.42** | 0/3 → 2/3 |

Nothing else moved: 13 torculus glyphs changed across both pages, the other
395 glyphs are bit-identical. McGill's residual is one badly degraded
`torculus33` whose strokes have merged — an ink problem, not a rule problem.

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
- `shape_from_class_name` — **landed**: the intervals were decoded from the
  classification name because the CSV has no row for it. Finds every glyph
  whose pitches depend on an unconfirmed reading (`neume.distropha`,
  `neume.clivis1`).
- Retained unchanged: `missing_staff`, `missing_clef`, `sparse_stave_lines`.

## Also in scope (found while investigating)

**Pitchless detection is membership-based, so unknown non-music classes get
pitches.** `neume_shapes.load_neume_shapes` only adds `divisio.*` /
`accidental.*` to `pitchless_classes` for rows *present in the CSV*.
`divisio.maior` and `staff` appear in real IC output but not the CSV, so
`is_pitchless` returns False and they fall through to the unknown-shape
fallback and receive a pitch. Make the test prefix-based, matching
`rodan_pitch_finder._PITCHLESS_PREFIXES`.

~~**Classes in real IC output but absent from the CSV.** `neume.clivis1` (18 on
McGill), `neume.distropha` (8), `neume.clivis2a`/`2b`, `neume.podatus2`,
`clef.f2`, `clef.g`, `neume.scandicus32` (in the cheatsheet images but not
the CSV). Handle by normalizing a trailing letter variant (`podatus2b` →
`podatus2`) before lookup, then falling back to the name-derived generator.~~
**Done** — see "Shape-table lookup fallbacks" above. (`clef.f2`/`clef.g` were
already covered separately by `_CLEF_NAME_RE`.)

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
| 0 | ~~`@intm` cumulative fix~~ **done**, with the torculus anchor rule it depends on. Still open: prefix-based pitchless test | generator-vs-CSV test (landed); `divisio.maior` no longer pitched (open) |
| 1 | `page_metrics.py`; anchor table; rewrite `_decompose` (geometry only) | per-class step diff vs Rodan collapses |
| 2 | Optional pixel refinement + rejection bound | refined vs geometric agree within a notehead |
| 3 | `span_mismatch` / clef snap / new flags; debug viz shows all components | flag counts on both sample pages |
| 4 | Regenerate sample outputs, three-way compare | debug images |

Phase 0 is independent of the redesign and worth landing on its own.

## Tests

- **Generator vs CSV** — ~~name-derived intervals equal CSV-derived intervals
  for all overlapping classes.~~ **Done**
  (`test_csv_intervals_match_intervals_derived_from_the_class_names`): all 27
  multi-note classes in the CSV, cross-checked against
  `intervals_from_class_name`, which shares no code with the parser. The test
  also asserts its own coverage, so a regex that stopped matching can't make
  it pass by checking nothing. This is the test that would have caught the
  `@intm` bug, and it guards it permanently.
- **Anchor unit tests** on synthetic staves — a virga whose bbox top sits on
  a line resolves to *that* line's step, not one step below; a clivis with a
  stem extending 3 steps below the lower head is unaffected by the stem.
- **Regression against the measured bias** — on `McGill_MS234-064` the median
  step difference vs Rodan for `virga` / `clivis2` / `podatus2b` must fall
  below ~0.35 steps, from today's −0.70 / +0.88 / −1.10.
- **No silent pitches** — `custos`, `divisio.maior`, and unknown classes are
  flagged or pitchless, never silently pitched.

## Open questions

Still open, and these need a musicologist rather than a heuristic:

1. **`custos` anchor.** 140 px (6.7 steps) tall with a nearly flat ink
   profile — a long stroke that dominates any centroid. Both current
   algorithms average the whole shape and agree with each other while almost
   certainly being wrong. Needs someone to state which end carries the pitch.
2. **`scandicus22a` / `22b` / `22c`** share intervals `[0, 1, 2]` but are
   drawn differently, and Rodan singles out `22b` for a bottom-left crop.
   Do they need three different anchor specs?
3. **Liquescents** — does the liquescent tail carry its own pitch, or is it
   ornamental? Affects whether `liquescent.*` is one component or two.
4. **`scale_unit` as a notehead-height fallback** — exact match on one page
   is not evidence. Confirm on more pages or drop the fallback.
5. **`CLEF_OCTAVE_REFERENCE`** (C=4/F=3/G=4) remains an unvalidated
   placeholder, unchanged by this plan.

### Closed

- ~~**`neume.clivis1`**~~ — not a valid class, and so not decoded. `clivisN` is
  a descent of an *N*th, so `clivis1` descends by nothing — and a ligature ties
  noteheads on *different* pitches; two on one pitch is a distropha. It is a
  classifier error, so the name-derived fallback refuses any digit of 1
  (`clivis1`, `podatus1`, `torculus12`, ...) rather than dressing the error up
  as a confident two-note reading. Those glyphs take the flagged single-note
  `approximate_unknown_shape` path and draw amber.
- ~~**`neume.distropha`'s unison**~~ — the neighbouring case, resolved the other
  way. It is a real class and its noteheads *are* a unison, so instead of being
  refused it collapses to the one pitch they share: `[0]`, one note, no flag.
- ~~**The `neume.clivis*` anchor**~~ — takes the top-head rule above. What
  settled it was not a better oracle but an assertion that needs none: on
  `clivis4b` the midpoint put the first note *outside the glyph's own bbox*,
  which is wrong however you measure. Capping the oracle's runs to one notehead
  then made the two-band measurement sound on both pages too, so the earlier
  "cannot be validated here" was the oracle's fault rather than the pages'.

  `neume.oblique*` is the one multi-note class still on the midpoint, and
  deliberately: it is a solid diagonal parallelogram whose heads share every
  column band, so no run split separates them — but that same symmetry means
  its whole-band centroid *is* the midpoint of its two notes. Forcing the
  clivis rule on it takes `oblique3`'s second note from 0.35 steps of error to
  0.95, and from 67% inside half a step to 0%.

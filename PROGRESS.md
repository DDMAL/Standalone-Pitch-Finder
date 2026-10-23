# Progress

Status as of 2026-10-16. The last commit on `main` is `d9c8aad` (2026-09-01).
Accuracy numbers here were recomputed on 2026-10-16 from the committed code and
labels, not copied from older docs.

## Current status

| Area | State |
|---|---|
| Algorithm #1 (`pitch_finder.py`) | Splits each neume into its notes and places them from one pixel-measured notehead. The default mode since July; unchanged since 2026-08-05. |
| Algorithm #2 (`rodan_pitch_finder.py`) | Reimplementation of Rodan's pitch finder, kept as an independent baseline. Not scored against the labels yet. |
| Ground truth | 2607 hand-labeled stave steps on 1873 glyphs across 13 Hufnagel pages. Stave steps only: there is no pitch-name or MEI ground truth, and no square-notation page is labeled. |
| Algorithm #1 accuracy | 81.6% of labeled notes (2127/2607) get the right stave step. 73.0% of glyphs (1367/1873) get every note right. Per-page numbers are in [ANNOTATION_PROGRESS.md](ANNOTATION_PROGRESS.md). Recompute them with `scripts/score_stave_steps.py`. |
| Stave-step CNN ([experiments/v1-hfngl](experiments/v1-hfngl/README.md)) | Single-note glyphs only. On the same 233 test glyphs, the classifier trained with augmentation beats algorithm #1: 93.6% vs 88.4% with hand-corrected staff lines, and 69.5% vs 55.8% with uncorrected ones. |
| Staff lines | On 5 of the 13 pages, staff-finding was wrong enough that the lines were fixed by hand (`fix_stafflines.py`). `staff_finding_rerun/` holds a fresh staff-finding run for all 13 pages. |
| Tests | 157 passing (`cd scripts && python -m pytest script_tests/`). |

## Open questions and next steps

- **Three pages score far below the rest, and their staff data is fine.** They are
  `Antiphonale_officii_Windeshemense_p009` (63.8% of notes), `NZ-Wt_MSR-03_065r`
  (67.5%) and `MS025b-01` (66.7% of notes, 38.2% of glyphs). On `MS025b-01` the main
  error is how many notes a glyph has, not where they are. On 117 of 272 labeled
  glyphs, algorithm #1 outputs a different number of notes than the annotator
  recorded. In 107 of those it outputs more, mostly for glyphs that IC classified as
  `neume.torculus33` (46) or `neume.clivis3a` (31). The likely cause is IC's
  classification on that page, not notehead finding. Windeshemense shows the same
  pattern on 28 of 145 glyphs. `065r` has not been examined yet.
- **A hand-fixed staff grouping is only used with `--no-regroup`.** `fix_stafflines.py`
  writes a `manually_grouped` flag, but no code reads it. If you run
  `run_pitch_finding.py` on one of the 5 fixed pages without `--no-regroup`, the
  stave grouping is rebuilt from the line geometry and the fix is lost without any
  warning. Reading the flag at load time would remove this trap.
- **Algorithm #2 has not been scored.** Scoring it against the same labels would give
  the first measured comparison between the two algorithms.
- **The CNN only handles single-note glyphs.** Multi-note neumes are still split by
  algorithm #1.
- **The square-notation folios are untouched.** Six are in `bbox-ground-truths/`;
  none has a page folder or labels yet.
- **The octave for each clef** (`CLEF_OCTAVE_REFERENCE`) is still a placeholder.

## Timeline

| Dates | Who | What |
|---|---|---|
| 07-21 | Kun Fang | First prototype: both algorithms (algorithm #1 placed notes from the bbox span), the parsers, the interval table, 52 tests |
| 07-29 to 08-05 | Yueqiao Zhang | Pixel anchoring, debug overlays, two-column regrouping, page-folder layout, `@intm` fix, first-notehead anchors, shape fallbacks, sample pages. Tests went from 52 to 147 |
| 08-20 | Yueqiao Zhang | CLAUDE.md, IIIF downloader |
| 08-20 to 08-24 | Kun Fang | Labeling GUI, labels for 13 pages, staff-line correction tools, 5 pages fixed by hand |
| 08-22 to 08-27 | Kun Fang | v1 stave-step CNN experiment |
| 09-01 | Kun Fang | Staff-finding rerun for the 13 pages, data provenance notes |
| 10-16 | Yueqiao Zhang | Scoring script for algorithm #1 against the labels, docs brought up to date |

All dates are in 2026.

## Work by Yueqiao Zhang

Commit hashes are listed for each item. The measured numbers are from the time of
the change and are recorded in more detail in those commit messages and in
[PLAN-neume-anchors.md](PLAN-neume-anchors.md).

### Finding the notehead in algorithm #1

The first version of algorithm #1 stretched each neume's notes evenly between the
top and bottom edges of its bbox. A virga's stem or a podatus's tail moves those
edges, which moves every note. On `McGill_MS234-064` one notehead is about 1.2
stave steps tall, so an error of half a notehead is enough to give a wrong pitch.

- **Pixel anchor mode** (`8a4ae91`, tests in `2d4a9fb`). The new default,
  `--anchor-mode pixel`, measures the ink in a crop of the glyph chosen per class,
  borrowing algorithm #2's crops, through the new `glyph_pixels.reference_point()`.
  `pitch_finder._anchor_interval()` uses the crop region to decide which note of the
  neume the measured point belongs to. The older `bbox` mode is still available. On
  McGill, the median per-class disagreement with algorithm #2 dropped from 0.49 to
  0.14 stave steps. On `GentAnt1475_0017_AC_rightcrop`, `bbox` mode was off by up to
  3.3 steps and pixel mode is within 0.34.
- **`@intm` fix** (`037abf6`). MEI gives each note's interval from the note before
  it, but the code read the intervals as offsets from the first note. That flipped
  the shape of every neume that changes direction: `torculus22` came out as
  `[0, 1, -1]` instead of `[0, 1, 0]`. A new test checks all 27 multi-note classes
  against intervals worked out from the class names alone.
- **Anchoring ligatures on their first notehead** (`037abf6`, `3222ad9`). For
  podatus, pescephalicus, scandicus and torculus, the anchor is the lowest notehead
  in the glyph's left column. For clivis it is the highest. The notehead is found
  from the ink itself: rows are weighed by how much ink they hold, so a staff line
  crossing the column does not count as a notehead. The anchor is limited to one
  notehead's height, measured by the new `notehead_height()`. Median error on
  McGill's podatus dropped from 0.43 to 0.16 steps. On GentAnt, a page none of the
  constants were tuned on, it dropped from 0.63 to 0.30. Error on `clivis4b` dropped
  from 1.08 to 0.08 steps. The center of every note on a multi-note glyph now falls
  inside that glyph's bbox (315 of 315).
- **Keeping algorithm #2 independent.** The new rules apply only to algorithm #1,
  through the `extended_rules` switch. Algorithm #2's output stayed byte-identical,
  so it is still a separate baseline to compare against.
- **Shape lookup fallbacks** (`3222ad9`). When a class has no exact row in the
  table, the lookup now tries three fallbacks in order:
  - drop a trailing variant letter (`clivis2a` becomes `clivis2`)
  - map repeated-note neumes to one pitch
  - work the intervals out from the class name

  The output records which source was used and flags anything below an exact match.
  On McGill, 46 glyphs that used to be reported as a single note are now split
  into their notes.

### Staff handling

- **Two-column regrouping** (`00248ad`, `staff_regroup.py`). Staff-finding groups
  staff lines by height only, so on a two-column page it merges the two columns'
  staves. The new module rebuilds the staves from the line geometry. On
  `Breviarium_ad_usum`, 13 staves that each mixed both columns became 20
  single-column staves, and the left column's pitch range went from f3–b4 to
  a3–f4. Single-column pages are not affected.
- **`Stave.y_at_step()`** (`1b54dd7`) converts a stave step back to a pixel row, for
  the overlays.

### Debug overlays (`1b54dd7`, `00248ad`)

- A crosshair at each computed notehead center, labeled `<note no>:<pitch>`.
- Staff lines drawn along their fitted curves.
- A second, unlabeled copy of each overlay, so labels never hide the markers.
- Label size that grows with the square root of the scale.
- Output paths checked before any rendering starts, and image writes that fail
  with an error instead of returning `False`.

### Inputs, layout and data

- **Page-folder layout.** `page_inputs.py` finds a page's three inputs and refuses
  to guess when there is more than one candidate (`00248ad`, `378ed5c`). Each page
  folder is split into `input/` and `output/` (`ab524b1`).
- **Sample pages.** Added `Breviarium_ad_usum` (two columns) and its crop
  (`d187ec4`), and `GentAnt1475_0017_AC_rightcrop` as a lower-quality page
  (`80e7390`). Corrected five IC classifications on Breviarium by hand, and switched
  McGill to the full 288-glyph page session (`5cc270b`).
- **IIIF downloader.** `tools/iiif_dl.py` downloads page images from a IIIF
  manifest (`351fc52`).

### Measuring against the labels

- **`scripts/score_stave_steps.py`** (2026-10-16) scores algorithm #1 against the
  hand labels for each page, with per-note and per-glyph results in separate
  columns. It detects hand-grouped staff files, and it warns when a page's IC XML
  no longer matches its labels. It was used to fix ANNOTATION_PROGRESS.md's
  accuracy column, which had mixed the two measures, and to find that MS025b-01's
  errors come mostly from wrong note counts.

### Documentation and tests

- Wrote `PLAN-neume-anchors.md` (design and measurements), the README sections on
  these features, and `CLAUDE.md` (`4e55960`). Updated the README, CLAUDE.md,
  ANNOTATION_PROGRESS.md and this file on 2026-10-16.
- Grew the test suite from 52 to 147 tests. Added `test_page_inputs.py` and
  `test_staff_regroup.py`, and extended the tests for `glyph_pixels`,
  `pitch_finder`, `neume_shapes` and `staff_io`.

## Work by Kun Fang

A short summary, so the split of work is clear. The commit history has the details.

- The first prototype: both algorithms, the parsers, the interval table and 52 tests.
- `annotate_notecenters.py` (the labeling GUI), the labels for all 13 pages, and
  `ANNOTATION_PROGRESS.md`.
- `fix_stafflines.py` and `check_stafflines.py`, and hand-fixed staff lines on 5
  pages. Rebuilt `MS025a-02` at the correct image scale. Fixed `staff_regroup` so
  that a zero-width staff line no longer crashes it.
- The `experiments/v1-hfngl` stave-step CNN.
- `bbox-ground-truths/` (Gen's 18 folios), `staff_finding_rerun/`, and the data
  provenance section of the README.

# MEI ground-truth check (v2-square)

Can the MEI files from `DDMAL/e2e-omr-resources` be used as ground truth for
the v2 square-notation experiment (one bbox and one pitch per note)? This
folder checks two manuscripts against their page images. **Both are usable.**
Einsiedeln needs its boxes taken from a different MEI version. MS73 needs its
zero-height boxes given a size.

## Sources

| | MEI (public) | Images (private HF dataset) |
|---|---|---|
| Einsiedeln (CH-E 611) | `resulting_mei_files/Einsiedeln/Reviewed no pitch correction needed/` + `.../Processed/` | `DDMAL-lab/All-Images`, `Pics Einsiedeln/Einsiedeln__Stiftsbibliothek__Codex_611_<folio>.jpg` |
| MS73 (CDN-Mlr MS73) | `resulting_mei_files/MS73/Processed/` (there is no Reviewed folder) | same dataset, `Pics MS73/`, named `MS73_<page>.jpg` or `_A_15th_century_Italian_antiphonal___manuscript__0_<page>_p.__0.jpg` |

The images are not stored in this repo. To download one (you need an HF token
with access to the DDMAL-lab org):

```bash
curl -sL -H "Authorization: Bearer $HF_TOKEN" -o <image dir>/<name> \
  "https://huggingface.co/datasets/DDMAL-lab/All-Images/resolve/main/Pics%20Einsiedeln/<name>"
```

## Layout

```
mei_groundtruth_check/
├── fix_zones.py              Einsiedeln: shared MEI parser + octave-drag fixes (a)/(b), compared
├── verify_bboxes.py          Einsiedeln: independent notehead detector + matching (also used for MS73)
├── make_manual_check.py      Einsiedeln: blind 200-tile manual spot-check page
├── check_reviewed_once.py    Einsiedeln: image check of Reviewed_once against the Reviewed set
├── visualize_einsiedeln.py   Einsiedeln: draw the Reviewed MEI as distributed
├── check_ms73.py             MS73: zero-height zones, octave test, box + pitch checks
├── visualize_ms73.py         MS73: draw the MEI (staff lines, boxes, pitches)
├── einsiedeln/
│   ├── mei/reviewed/         10 folios as distributed (octave-dragged)
│   ├── mei/processed/        the same 10 before the drag
│   ├── groundtruth_viz/      Reviewed MEI drawn on the page (+ octave_drag_evidence.png)
│   ├── fix_comparison/       option (a) vs (b) overlays, per_element.csv
│   ├── bbox_verification/    notehead check overlays, per_note.csv, summary.csv
│   ├── manual_check/         manual_check.html, key.csv, manual_check_results.csv (human verdicts)
│   └── reviewed_once_check/  notehead check overlays for 10 + 10 folios, summary.csv
└── ms73/
    ├── mei/                  10 sampled pages (Processed)
    ├── groundtruth_viz/      4 pages drawn (004, 110, 230, 289)
    └── verify/               notehead check overlays, all 10 pages
```

Folios: Einsiedeln 081r 085r 095r 125r 128r 165r 180r 200r 230r 260r. MS73
pages 004 048 079 110 140 170 200 230 260 289, spread evenly over the 286
pages in the repo.

## Einsiedeln

- **The note boxes in Reviewed are shifted up by 7 diatonic steps.** Rodan/Neon
  first read C clefs as C3. Reviewers fixed the pitches by dragging neumes up an
  octave in Neon, and that moved their zones up as well. Clefs and lyrics were
  not dragged, and staves under an F clef were not dragged either.
- **Fix used: option (a).** Take each note's zone from `Processed`, matched by
  `xml:id`, and keep its pitch from `Reviewed`. On 5,512 notes the detector
  marks 77.5% as on a notehead, compared with 9.1% for the boxes as
  distributed. Option (b), shifting down 7 steps, gives 73.2%.
- **Manual blind check** (200 tiles, `manual_check/`): estimated 6.4% ± 4.0%
  of boxes wrong overall. Notes outside ligatures are 1.3% ± 0.8% wrong. In
  ligatures about 66% are wrong: Neon draws the second note of a ligature
  directly under the first instead of at the end of the oblique stroke. Every
  box also sits about 0.45 step below its notehead, uniformly.
- **Decision:** use all notes, ligated ones included, with option (a) boxes and
  no offset correction.
- `divLine` zones are unreliable in every version and are not needed.
- **`Reviewed_once` (264 more folios, mostly 001r–080v) is usable too.** It was
  reviewed before Neon's C-clef fix, so notes under a C clef are encoded an
  octave low. But it was never octave-dragged: 98% of its note zones are
  identical to `Processed`. Checked on 10 folios against the images
  (`check_reviewed_once.py`), its own boxes do at least as well as option (a)
  on 10 Reviewed folios: 80.1% vs 77.5% verified, and 95.7% vs 92.6% of
  matched noteheads on the staff position their pitch gives (reading C as C3),
  none off by an octave. The one weak folio, 001r (35.6%), is faded and
  stained; with `local_bg=True` it reaches 80.8%. One folio, 138v, already has
  its octaves raised (its C clefs carry `dis="8"`), so the dataset build reads
  each page's convention off its own geometry.

## MS73

- **Coordinates are half the image size** (`Processed`; `Processed and resized`
  is the 2× version). The scripts scale the image down to the MEI `<surface>`
  size first.
- **No octave problem.** Clefs and notes follow the correct C4 convention
  (offset ≈ 0 steps).
- **The MEI staff lines match the real ones** (the staff lines are red ink),
  to within 0.1–0.3 step. The remaining gap comes from page curvature against
  straight MEI lines.
- **The pitch labels match the image.** 88.7% of matched noteheads sit within
  half a step of where their pitch puts them on the MEI staff (758 notes). None
  is off by an octave.
- **57% of note zones have zero height.** Given the page's median note size,
  anchored at the recorded top-left corner, they land on noteheads as often as
  the boxes that have a height (68.1% vs 62.4% verified).
- A few staves have every box offset by about one step (6 of 89 staff groups,
  all on pages 260 and 289).
- The noteheads are often faint, so `check_ms73.py` runs the detector with
  `local_bg=True`. That thresholds contrast against the local paper colour
  inside the staff bands. Einsiedeln runs without it, and its results are
  unchanged.
- **Verdict:** usable.

## Re-running

```bash
conda activate umil   # needs cv2, numpy, scipy
python fix_zones.py            --image-dir <Einsiedeln images>
python verify_bboxes.py        --image-dir <Einsiedeln images>   # before make_manual_check.py
python make_manual_check.py    --image-dir <Einsiedeln images>
python check_reviewed_once.py  # images from <repo>/data/images/einsiedeln (../dataset/fetch_sources.py --images)
python visualize_einsiedeln.py --image-dir <Einsiedeln images> [folio ...]
python check_ms73.py           --image-dir <MS73 images>
python visualize_ms73.py       --image-dir <MS73 images> [page ...]
```

All outputs are deterministic. `make_manual_check.py` rewrites
`manual_check.html` and `key.csv` (fixed seed) but never touches
`manual_check_results.csv`. The PNG overlays add up to about 350 MB; they can
be regenerated and probably should not all be committed.

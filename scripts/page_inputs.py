"""
Find a page's three inputs (image, IC XML, staff-finding JSON) from its folder.

Every page in this repo is one folder holding those three files, so naming the
folder is enough -- the files' actual names differ per page (`ic.xml` vs
`ic-session-<page>-page.xml`, `original_crop.jpg` vs `<page>.jpg`) and spelling
all three out on the command line is the bulk of the typing.

A page folder keeps its inputs in `input/` and takes its artifacts in
`output/`; discovery searches `input/` and never looks at `output/`, so a run's
own artifacts can't become candidate inputs for the next one (this used to need
the _DERIVED_MARKERS name filter below, which now only matters for folders
still in the old flat layout, where inputs and artifacts share one directory).

Discovery is deliberately strict: each kind must resolve to exactly one
candidate or it raises, naming what it found and which flag overrides it. A
wrong guess here would be silently attributed to the algorithm ("why is every
pitch off?") when the real cause was a stale IC XML picked up next door.
"""

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

# Extensions treated as page images, both here (which file is the scan) and by
# viz_utils (which extensions cv2 can pick an encoder for). Lives in this
# module so path resolution stays importable without opencv installed.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})

# Artifact names produced *by* this repo (and by staff-finding / IC), which sit
# in the same folder as the page image and would otherwise be candidates for
# it. Matched as substrings of the stem, lowercased.
_DERIVED_MARKERS = ("debug", "stave_grouping", "pitch_finding", "predicted",
                    "overlay", "nolabels")

# The per-page subfolders: inputs are read from one, artifacts written to the
# other. Both are conventions of the page folder, not of any single CLI, so
# every driver lands its artifacts in the same place for a given page.
INPUT_DIR_NAME = "input"
OUTPUT_DIR_NAME = "output"


@dataclass
class PageInputs:
    """The three per-page inputs, plus the folders around them.

    page_dir is the page folder; input_dir is where the three files were
    actually found (page_dir/input/ when that exists, else page_dir itself, for
    folders still in the flat layout) and output_dir is where callers should
    write artifacts. output_dir need not exist yet -- it is created at write
    time, so a usage error doesn't leave an empty folder behind.
    """
    image: Path
    ic_xml: Path
    staff_json: Path
    page_dir: Path
    input_dir: Path
    output_dir: Path


def resolve_page_inputs(page: Path, image: Path = None, ic_xml: Path = None,
                        staff_json: Path = None) -> PageInputs:
    """Resolve a page folder (or any file inside one) into a PageInputs.

    page may be the page folder, its input/ folder, or a file in either --
    passing the image is the natural way to disambiguate a folder holding two
    pages, and it reads the same as the old --image, so
    `run_pitch_finding.py page_dir/input/page.jpg` works too. All of those name
    the same page, so all of them get the same output_dir.

    Explicitly passed image/ic_xml/staff_json win over discovery and are used
    as-is (they may point outside page's folder); each is checked for existence
    here so a typo is reported before any parsing starts.
    """
    page = Path(page)
    if page.is_dir():
        page_dir = page
    elif page.exists():
        page_dir = page.parent
        if image is None and page.suffix.lower() in IMAGE_SUFFIXES:
            image = page
    else:
        raise ValueError(f"No such page folder or file: {page}")

    # Naming input/ (or a file inside it) is naming the page it belongs to:
    # step up so artifacts don't land in input/output/.
    if page_dir.name == INPUT_DIR_NAME:
        page_dir = page_dir.parent
    # Folders predating the input/ convention keep their three files at the top
    # level; search there so they still resolve.
    input_dir = page_dir / INPUT_DIR_NAME
    if not input_dir.is_dir():
        input_dir = page_dir

    image = _pick(input_dir, image, "image", _image_candidates(input_dir), "--image")
    # Narrowing by the image's stem is what makes a two-page folder work once
    # the image has been named: `<page>_stafflines.json` belongs to `<page>.jpg`.
    return PageInputs(
        image=image,
        ic_xml=_pick(input_dir, ic_xml, "IC XML",
                     _prefer_stem(_find(input_dir, "*.xml"), image), "--ic-xml"),
        staff_json=_pick(input_dir, staff_json, "staff-finding JSON",
                         _prefer_stem(_find(input_dir, "*stafflines*.json"), image),
                         "--staff-json"),
        page_dir=page_dir,
        input_dir=input_dir,
        output_dir=page_dir / OUTPUT_DIR_NAME,
    )


def _find(input_dir: Path, pattern: str) -> list[Path]:
    """Files whose name matches pattern (case-insensitively) in input_dir.

    Top level first, then recursively: the top-level pass keeps a page's own
    `ic.xml` from ever losing to a copy nested in an export folder, and the
    recursive pass is what finds a flat-layout page's
    `ic_output/ic-session-*.xml`, where nothing matches at the top level at all.
    """
    for scope in (input_dir.glob("*"), input_dir.rglob("*")):
        hits = sorted(p for p in scope
                      if p.is_file() and fnmatch(p.name.lower(), pattern.lower()))
        if hits:
            return hits
    return []


def _image_candidates(input_dir: Path) -> list[Path]:
    """Top-level images in input_dir that aren't previously rendered artifacts.

    Non-recursive on purpose: a page folder can contain whole subfolders of
    derived images (IC's `ic_input/`, this repo's `test/`), and a recursive
    search would turn a one-image page into an ambiguity every time.
    """
    return sorted(p for p in input_dir.glob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                  and not _is_derived(p))


def _is_derived(path: Path) -> bool:
    stem = path.stem.lower()
    return any(marker in stem for marker in _DERIVED_MARKERS)


def _prefer_stem(candidates: list[Path], image: Path) -> list[Path]:
    """Keep only the candidates named after image, if any are."""
    if len(candidates) < 2:
        return candidates
    named = [p for p in candidates if image.stem.lower() in p.name.lower()]
    return named or candidates


def _pick(input_dir: Path, explicit: Path, kind: str, candidates: list[Path],
          flag: str) -> Path:
    """Return the explicit path if given, else the one candidate, else raise."""
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.is_file():
            raise ValueError(f"{flag}: no such file: {explicit}")
        return explicit

    if not candidates:
        raise ValueError(f"Found no {kind} in {input_dir}; pass one with {flag}.")
    if len(candidates) > 1:
        names = ", ".join(str(p.relative_to(input_dir)) for p in candidates)
        raise ValueError(f"Found {len(candidates)} candidate {kind} files in "
                         f"{input_dir} ({names}); pick one with {flag}.")
    return candidates[0]

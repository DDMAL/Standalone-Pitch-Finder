"""
Find a page's three inputs (image, IC XML, staff-finding JSON) from its folder.

Every page in this repo is one folder holding those three files plus whatever
artifacts previous runs left behind, so naming the folder is enough -- the
files' actual names differ per page (`ic.xml` vs
`ic_output/ic-session-<page>-page.xml`, `original.jpeg` vs `<page>.jpg`) and
spelling all three out on the command line is the bulk of the typing.

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


@dataclass
class PageInputs:
    """The three per-page inputs, plus the folder they were found in."""
    image: Path
    ic_xml: Path
    staff_json: Path
    page_dir: Path


def resolve_page_inputs(page: Path, image: Path = None, ic_xml: Path = None,
                        staff_json: Path = None) -> PageInputs:
    """Resolve a page folder (or any file inside one) into a PageInputs.

    page may be the folder itself or a file in it -- passing the image is the
    natural way to disambiguate a folder holding two pages, and it reads the
    same as the old --image, so `run_pitch_finding.py page.jpg` works too.

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

    image = _pick(page_dir, image, "image", _image_candidates(page_dir), "--image")
    # Narrowing by the image's stem is what makes a two-page folder work once
    # the image has been named: `<page>_stafflines.json` belongs to `<page>.jpg`.
    return PageInputs(
        image=image,
        ic_xml=_pick(page_dir, ic_xml, "IC XML",
                     _prefer_stem(_find(page_dir, "*.xml"), image), "--ic-xml"),
        staff_json=_pick(page_dir, staff_json, "staff-finding JSON",
                         _prefer_stem(_find(page_dir, "*stafflines*.json"), image),
                         "--staff-json"),
        page_dir=page_dir,
    )


def _find(page_dir: Path, pattern: str) -> list[Path]:
    """Files whose name matches pattern (case-insensitively) in page_dir.

    Top level first, then recursively: the top-level pass keeps a page's own
    `ic.xml` from ever losing to a copy nested in an export folder, and the
    recursive pass is what finds GentAnt's `ic_output/ic-session-*.xml`, where
    nothing matches at the top level at all.
    """
    for scope in (page_dir.glob("*"), page_dir.rglob("*")):
        hits = sorted(p for p in scope
                      if p.is_file() and fnmatch(p.name.lower(), pattern.lower()))
        if hits:
            return hits
    return []


def _image_candidates(page_dir: Path) -> list[Path]:
    """Top-level images in page_dir that aren't previously rendered artifacts.

    Non-recursive on purpose: a page folder can contain whole subfolders of
    derived images (IC's `ic_input/`, this repo's `test/`), and a recursive
    search would turn a one-image page into an ambiguity every time.
    """
    return sorted(p for p in page_dir.glob("*")
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


def _pick(page_dir: Path, explicit: Path, kind: str, candidates: list[Path],
          flag: str) -> Path:
    """Return the explicit path if given, else the one candidate, else raise."""
    if explicit is not None:
        explicit = Path(explicit)
        if not explicit.is_file():
            raise ValueError(f"{flag}: no such file: {explicit}")
        return explicit

    if not candidates:
        raise ValueError(f"Found no {kind} in {page_dir}; pass one with {flag}.")
    if len(candidates) > 1:
        names = ", ".join(str(p.relative_to(page_dir)) for p in candidates)
        raise ValueError(f"Found {len(candidates)} candidate {kind} files in "
                         f"{page_dir} ({names}); pick one with {flag}.")
    return candidates[0]

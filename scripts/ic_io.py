"""
IC (Interactive Classifier) glyph database XML I/O.

Parses the Gamera glyph-database XML produced by the IC classification step:
one <glyph> element per bbox, carrying its page-pixel bounding box and its
classification (name + confidence + state). Only the fields pitch-finding
needs are extracted; feature vectors and raw data blobs are ignored.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass
class Glyph:
    """One glyph from the IC XML.

    Attributes:
        index: Position of this glyph in the XML (stable ordering for
            referencing glyphs from output/debug artifacts).
        ulx, uly: Upper-left corner of the bounding box, in page pixels.
        nrows, ncols: Height and width of the bounding box, in page pixels.
        class_name: Gamera classification id, e.g. "neume.podatus3",
            "clef.c", "skip.dot", "text".
        confidence: Classifier confidence for class_name. 0.0 for glyphs
            that were never actually run through the neume classifier
            (state == "UNCLASSIFIED").
        state: "AUTOMATIC" (classified) or "UNCLASSIFIED" (text bbox
            carried through without classification).
    """
    index: int
    ulx: int
    uly: int
    nrows: int
    ncols: int
    class_name: str
    confidence: float
    state: str

    @property
    def center_x(self) -> float:
        return self.ulx + self.ncols / 2

    @property
    def center_y(self) -> float:
        return self.uly + self.nrows / 2

    @property
    def lry(self) -> int:
        return self.uly + self.nrows


def parse_ic_xml(path: Path) -> list[Glyph]:
    """Parse a Gamera glyph-database XML file into a list of Glyphs.

    Malformed <glyph> elements (missing ids/id or bbox attributes) are
    skipped with a printed warning rather than raising, matching the
    permissive-parsing convention used by staff_io.py.
    """
    tree = ET.parse(path)
    glyphs = []
    for index, glyph_el in enumerate(tree.getroot().findall(".//glyph")):
        try:
            ids_el = glyph_el.find("ids")
            id_el = ids_el.find("id")
            glyphs.append(
                Glyph(
                    index=index,
                    ulx=int(glyph_el.get("ulx")),
                    uly=int(glyph_el.get("uly")),
                    nrows=int(glyph_el.get("nrows")),
                    ncols=int(glyph_el.get("ncols")),
                    class_name=id_el.get("name"),
                    confidence=float(id_el.get("confidence")),
                    state=ids_el.get("state"),
                )
            )
        except (AttributeError, TypeError, ValueError) as e:
            print(f"  Skipping malformed <glyph> at index {index} in {path}: {e}")
    return glyphs

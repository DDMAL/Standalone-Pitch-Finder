"""Sanity checks for IC XML parsing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_io import parse_ic_xml

SAMPLE_XML = """<?xml version='1.0' encoding='utf-8'?>
<gamera-database version="2.0">
  <glyphs>
    <glyph uly="432" ulx="318" nrows="22" ncols="11">
      <ids state="AUTOMATIC">
        <id name="neume.inclinatum" confidence="0.156548"/>
      </ids>
      <data>irrelevant</data>
      <features scaling="1.0" version="ic-core/v1"></features>
    </glyph>
    <glyph uly="1126" ulx="470" nrows="29" ncols="78">
      <ids state="UNCLASSIFIED">
        <id name="text" confidence="0.000000"/>
      </ids>
    </glyph>
  </glyphs>
</gamera-database>
"""


def test_parse_ic_xml(tmp_path):
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(SAMPLE_XML)

    glyphs = parse_ic_xml(xml_path)
    assert len(glyphs) == 2

    g0 = glyphs[0]
    assert g0.index == 0
    assert (g0.ulx, g0.uly, g0.ncols, g0.nrows) == (318, 432, 11, 22)
    assert g0.class_name == "neume.inclinatum"
    assert g0.state == "AUTOMATIC"
    assert g0.confidence > 0
    assert g0.center_x == 318 + 11 / 2
    assert g0.lry == 432 + 22

    g1 = glyphs[1]
    assert g1.state == "UNCLASSIFIED"
    assert g1.class_name == "text"
    assert g1.confidence == 0.0


def test_skips_malformed_glyph(tmp_path, capsys):
    bad_xml = """<?xml version='1.0'?>
<gamera-database version="2.0">
  <glyphs>
    <glyph uly="10" ulx="10" nrows="5" ncols="5">
      <ids state="AUTOMATIC"></ids>
    </glyph>
    <glyph uly="20" ulx="20" nrows="5" ncols="5">
      <ids state="AUTOMATIC">
        <id name="neume.punctum" confidence="0.9"/>
      </ids>
    </glyph>
  </glyphs>
</gamera-database>
"""
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text(bad_xml)

    glyphs = parse_ic_xml(xml_path)
    assert len(glyphs) == 1
    assert glyphs[0].class_name == "neume.punctum"

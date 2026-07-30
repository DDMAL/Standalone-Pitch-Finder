"""Discovery rules for a page folder's (image, IC XML, staff JSON) triple."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from page_inputs import resolve_page_inputs


def make_page(dir_path: Path, image="page.jpg", ic_xml="ic.xml",
              staff_json="page_stafflines.json", extras=()):
    dir_path.mkdir(parents=True, exist_ok=True)
    for name in (image, ic_xml, staff_json, *extras):
        path = dir_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    return dir_path


def test_finds_the_three_inputs(tmp_path):
    page_dir = make_page(tmp_path / "page")
    inputs = resolve_page_inputs(page_dir)
    assert inputs.image == page_dir / "page.jpg"
    assert inputs.ic_xml == page_dir / "ic.xml"
    assert inputs.staff_json == page_dir / "page_stafflines.json"
    assert inputs.page_dir == page_dir


def test_ignores_artifacts_from_previous_runs(tmp_path):
    # A folder that's been run before holds several images; only the scan is
    # an input. Without this, running twice in a row would break.
    page_dir = make_page(tmp_path / "page", extras=(
        "page_pitch_finding_debug.jpg", "page_rodan_pitch_finding_debug.jpg",
        "page_pitch_finding_debug_nolabels.jpg", "page_ic_debug.jpg",
        "page_stave_grouping_hq.png", "page_pitch_finding.json"))
    assert resolve_page_inputs(page_dir).image == page_dir / "page.jpg"


def test_ignores_images_in_subfolders(tmp_path):
    # IC's ic_input/ and this repo's test/ hold derived images under names that
    # don't say so; a recursive image search would make every page ambiguous.
    page_dir = make_page(tmp_path / "page",
                         extras=("ic_input/page_predicted.jpg", "test/test_new.jpg"))
    assert resolve_page_inputs(page_dir).image == page_dir / "page.jpg"


def test_finds_ic_xml_nested_in_an_export_folder(tmp_path):
    # GentAnt's layout: nothing matches *.xml at the top level.
    page_dir = tmp_path / "page"
    make_page(page_dir, ic_xml="ic_output/ic-session-page.xml")
    assert resolve_page_inputs(page_dir).ic_xml == page_dir / "ic_output/ic-session-page.xml"


def test_top_level_ic_xml_beats_a_nested_one(tmp_path):
    page_dir = make_page(tmp_path / "page", extras=("ic_output/ic-session-old.xml",))
    assert resolve_page_inputs(page_dir).ic_xml == page_dir / "ic.xml"


def test_only_stafflines_json_counts_as_the_staff_json(tmp_path):
    # A page folder also holds IC's input JSON and our own output JSON.
    page_dir = make_page(tmp_path / "page",
                         extras=("page.json", "all_predictions.json"))
    assert resolve_page_inputs(page_dir).staff_json == page_dir / "page_stafflines.json"


def test_passing_the_image_resolves_its_folder(tmp_path):
    page_dir = make_page(tmp_path / "page")
    assert resolve_page_inputs(page_dir / "page.jpg").ic_xml == page_dir / "ic.xml"


def test_passing_the_image_disambiguates_two_pages_in_one_folder(tmp_path):
    page_dir = make_page(tmp_path / "pages", extras=(
        "other.jpg", "ic-session-other.xml", "other_stafflines.json"))
    inputs = resolve_page_inputs(page_dir / "other.jpg")
    assert inputs.ic_xml == page_dir / "ic-session-other.xml"
    assert inputs.staff_json == page_dir / "other_stafflines.json"


def test_ambiguous_image_is_an_error_naming_the_candidates(tmp_path):
    page_dir = make_page(tmp_path / "pages", extras=("other.jpg",))
    with pytest.raises(ValueError, match=r"2 candidate image files.*--image"):
        resolve_page_inputs(page_dir)


def test_missing_input_is_an_error_naming_its_flag(tmp_path):
    page_dir = tmp_path / "page"
    page_dir.mkdir()
    (page_dir / "page.jpg").write_text("")
    with pytest.raises(ValueError, match=r"no IC XML.*--ic-xml"):
        resolve_page_inputs(page_dir)


def test_explicit_paths_override_discovery(tmp_path):
    page_dir = make_page(tmp_path / "page")
    elsewhere = make_page(tmp_path / "elsewhere", ic_xml="fixed.xml")
    inputs = resolve_page_inputs(page_dir, ic_xml=elsewhere / "fixed.xml")
    assert inputs.ic_xml == elsewhere / "fixed.xml"
    assert inputs.image == page_dir / "page.jpg"  # the rest still discovered


def test_explicit_path_that_does_not_exist_is_an_error(tmp_path):
    page_dir = make_page(tmp_path / "page")
    with pytest.raises(ValueError, match="--staff-json: no such file"):
        resolve_page_inputs(page_dir, staff_json=tmp_path / "nope.json")


def test_nonexistent_page_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No such page folder or file"):
        resolve_page_inputs(tmp_path / "nope")

"""Same figure as demo_examples.py, but for every glyph in the 247-example
held-out test set instead of a curated handful, paginated into multiple
images since one image with 247 rows isn't viewable. Exploratory/on-demand,
not meant to be regenerated on every run.

    python demo_examples_all.py
"""
from pathlib import Path

from evaluate import get_test_predictions, uncorrected_crops
from demo_examples import render_grid

HERE = Path(__file__).resolve().parent
ROWS_PER_PAGE = 20


def main():
    test_rows, y_test, preds = get_test_predictions()
    X_test_u, valid_u, tops_u, bottoms_u, overridden_u = uncorrected_crops(test_rows)

    n = len(test_rows)
    n_pages = (n + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
    print(f"rendering all {n} test glyphs across {n_pages} pages ({ROWS_PER_PAGE}/page)")
    for page_i in range(n_pages):
        idx = list(range(page_i * ROWS_PER_PAGE, min((page_i + 1) * ROWS_PER_PAGE, n)))
        out_path = HERE / "figures" / f"demo_examples_all_page{page_i+1}.png"
        render_grid(test_rows, y_test, preds, X_test_u, tops_u, bottoms_u, overridden_u, idx, out_path)


if __name__ == "__main__":
    main()

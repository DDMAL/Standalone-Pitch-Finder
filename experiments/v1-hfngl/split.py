"""One held-out test split, built per-page and per-class so no single large
page or common class dominates it. Shared by train.py and evaluate.py --
same seed, same logic, so both see the identical train/test division.
"""
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42
TEST_FRAC = 0.2
MIN_PAGE_FOR_TEST = 5  # pages with fewer real single-note labels go entirely to train


def build_split(rows, log=print):
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)

    train_rows, test_rows = [], []
    for page, page_rows in by_page.items():
        if len(page_rows) < MIN_PAGE_FOR_TEST:
            train_rows += page_rows
            log(f"  {page}: n={len(page_rows)} < {MIN_PAGE_FOR_TEST} -> all train, 0 test")
            continue
        classes = np.array([r["class_name"] for r in page_rows])
        counts = {c: (classes == c).sum() for c in set(classes)}
        strat_key = np.array([c if counts[c] >= 2 else "__rare__" for c in classes])
        idx = np.arange(len(page_rows))
        try:
            tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC, random_state=SEED, stratify=strat_key)
        except ValueError:
            tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC, random_state=SEED)
        train_rows += [page_rows[i] for i in tr_idx]
        test_rows += [page_rows[i] for i in te_idx]
        log(f"  {page}: n={len(page_rows)} -> train={len(tr_idx)} test={len(te_idx)}")

    return train_rows, test_rows

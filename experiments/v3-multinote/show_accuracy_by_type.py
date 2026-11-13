"""Figures: accuracy by neume type and by number of notes.

A neume's type is read off its labels: the direction from each note to the
next (up / down / same), named after the usual square-notation neume where
there is one (u = podatus, d = clivis, ud = torculus, du = porrectus, uu =
scandicus, dd = climacus, ...). Neumes of 5 or more notes are pooled, as are those
with two neighbouring notes on the same step (a repeated pitch); types with fewer
than MIN_N test neumes go to "rare shapes".

  figures/acc_by_type_test.png       test set: step 2 (final neume model) next to step 1
                                     (every note on its own box, cls_aug_bbox128_shift15_all)
  figures/acc_by_type_errors.png     test set, final model: note count / notes / whole neume
  figures/acc_by_type_salzinnes.png  Salzinnes: the neume model before / after spacing jitter
  figures/acc_by_size.png            whole neume right by number of notes, test and Salzinnes
  --compare M1 M2 ...: figures/acc_by_type_compare_{val,test,salzinnes}.png, whole neume
                       right by type for several neume models (choose on val; report test
                       and Salzinnes)

    python show_accuracy_by_type.py   # needs results/notes_predictions.npz (eval_notes.py)
    python show_accuracy_by_type.py --compare neume_shift15_hjit_hscale neume_pairs_w03 neume_c16
"""
import collections

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paths import FIGURES  # noqa: E402
from eval_neume import step1_notes_right  # noqa: E402
from neume_data import CROPS_CTX, geom_of, load_neumes, predict_neumes  # noqa: E402
from neume_model import MAX_NOTES  # noqa: E402
from train import device, predict_steps  # noqa: E402  (v2's train.py)
from train_neume import CONFIGS, FINAL, load_model  # noqa: E402
from train_notes import load_model as load_note_model  # noqa: E402

BEFORE = "neume_shift15_hjit"           # the neume model without spacing jitter
STEP1 = "cls_aug_bbox128_shift15_all"   # step 1: every note on its own box
MIN_N = 40
NAMES = {
    "": "single note", "u": "podatus (up)", "d": "clivis (down)", "s": "two notes, same step",
    "ud": "torculus (up-down)", "du": "porrectus (down-up)", "uu": "scandicus (up-up)", "dd": "climacus (down-down)",
    "udd": "pes subbipunctis (up-down-down)", "ddd": "climacus (down x3)", "dud": "porrectus flexus (down-up-down)",
    "udu": "torculus resupinus (up-down-up)", "uud": "scandicus flexus (up-up-down)", "duu": "down-up-up",
    "uuu": "up x3", "dduu": "down-down-up-up",
}
C1, C2, C3 = "#3a6fc4", "#e0892f", "#8a8f99"


REPEAT = "with a repeated step"  # two neighbouring notes on one step
RARE = "other rare shapes"


def type_of(steps, n):
    if n >= 5:
        return "5 or more notes"
    s = [int(v) for v in steps[:n]]
    key = "".join("u" if b > a else "d" if b < a else "s" for a, b in zip(s, s[1:]))
    if "s" in key:
        return REPEAT
    return NAMES.get(key, RARE)


def types(d):
    return np.array([type_of(st, n) for st, n in zip(d["steps"], d["n_notes"])])


def scores(d, steps, n):
    """Per neume: count right, notes right (fraction), whole neume right."""
    has = np.arange(MAX_NOTES)[None] < d["n_notes"][:, None]
    ok = has & (steps == d["steps"])
    count = n == d["n_notes"]
    return count, ok.sum(1) / d["n_notes"], np.all(ok == has, 1) & count


def order(t, n_notes, min_n=MIN_N):
    """Types with at least min_n neumes, by number of notes, then frequency."""
    c = collections.Counter(t)
    nn = {k: int(min(n_notes[t == k].min(), 5)) for k in c}
    last = {REPEAT: 1, RARE: 2}
    return sorted((k for k, v in c.items() if v >= min_n), key=lambda k: (last.get(k, 0), nn[k], -c[k])), c


def pool(t, n_notes, min_n):
    """Types with fewer than min_n neumes become RARE; RARE and REPEAT go last."""
    c = collections.Counter(t)
    return np.array([RARE if c[k] < min_n and k not in (REPEAT,) else k for k in t])


def hbars(ax, labels, series, counts, xlim=(0, 100), title=""):
    """Horizontal grouped bars: series = [(name, values in %, colour)]; n written next to each label."""
    y = np.arange(len(labels))[::-1]
    h = 0.8 / len(series)
    for k, (name, vals, col) in enumerate(series):
        yy = y + (len(series) - 1) / 2 * h - k * h
        ax.barh(yy, vals, height=h * 0.92, color=col, label=name)
        for yi, v in zip(yy, vals):
            ax.text(min(v, xlim[1]) + 0.6, yi, f"{v:.1f}", va="center", fontsize=8.5)
    ax.set_yticks(y, [f"{l}  (n={counts[l]:,})" for l in labels], fontsize=10)
    ax.set_xlim(*xlim)
    ax.set_xlabel("%")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series), fontsize=9, frameon=False)
    ax.set_title(title, loc="left", fontsize=12, pad=26)


def predict(name, d, dev):
    m = load_model(name, dev)
    return predict_neumes(m, torch.tensor(d["crops"], device=dev), dev, CONFIGS[name], geom_of(d, dev))


def salzinnes(dev):
    import eval_salzinnes as E
    neumes, _ = E.load_pages()
    d = {k: np.array([x[k] for x in neumes]) for k in ("steps", "n_notes", "box_l", "box_r", "unit_px")}
    d["crops"] = np.stack([x["ctx"] for x in neumes])
    m, _ = load_note_model(STEP1, dev)
    pred = predict_steps(m, torch.tensor(np.concatenate([np.stack(x["note_crops"]) for x in neumes]), device=dev), True, dev)
    owner = np.repeat(np.arange(len(neumes)), d["n_notes"])
    slot = np.concatenate([np.arange(k) for k in d["n_notes"]])
    ok = np.zeros((len(neumes), MAX_NOTES), bool)
    ok[owner, slot] = pred == d["steps"][owner, slot]
    has = np.arange(MAX_NOTES)[None] < d["n_notes"][:, None]
    return d, np.all(ok == has, 1)


def compare(models, dev):
    colors = [C1, C2, "#5aa469", "#9b59b6", C3]
    for split in ("val", "test", "salzinnes"):
        if split == "salzinnes":
            d, _ = salzinnes(dev)
        else:
            d = load_neumes(split, crops_dir=CROPS_CTX)
        t = pool(types(d), d["n_notes"], 10 if split == "salzinnes" else MIN_N)
        labels, counts = order(t, d["n_notes"], 1)
        series, overall = [], []
        for k, name in enumerate(models):
            _, _, whole = scores(d, *predict(name, d, dev))
            multi = d["n_notes"] > 1
            overall.append(f"{name}: multi-note {100 * whole[multi].mean():.1f}%, single-note {100 * whole[~multi].mean():.1f}%")
            series.append((name, [100 * float(np.mean(whole[t == l])) for l in labels], colors[k]))
        fig, ax = plt.subplots(figsize=(11, 0.36 * len(labels) * len(models) + 2.2))
        title = {"val": "Validation set", "test": "Test set", "salzinnes": "Salzinnes (unseen manuscript)"}[split]
        hbars(ax, labels, series, counts, (0, 106), f"{title}: neumes entirely right, by type")
        fig.text(0.01, 0.005, ";  ".join(overall), fontsize=8.5, color="#444")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(FIGURES / f"acc_by_type_compare_{split}.png", dpi=130)
        plt.close(fig)
        print(split, "|", " | ".join(overall))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()
    dev = device()
    if args.compare:
        compare(args.compare, dev)
        return
    d = load_neumes("test", crops_dir=CROPS_CTX)
    t = types(d)
    cnt, notes, whole = scores(d, *predict(FINAL, d, dev))
    has = np.arange(MAX_NOTES)[None] < d["n_notes"][:, None]
    s1_whole = np.all(step1_notes_right(d)["cls_aug_bbox128_shift15_all"] == has, 1)
    t = pool(t, d["n_notes"], MIN_N)
    labels, counts = order(t, d["n_notes"], 1)
    pct = lambda v: [100 * float(np.mean(v[t == l])) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(labels) + 1.6))
    hbars(ax, labels, [("one crop per neume (final model)", pct(whole), C1),
                       ("one crop per note (step 1)", pct(s1_whole), C3)],
          counts, (0, 106), "Test set (Einsiedeln + MS73): neumes entirely right, by type")
    fig.tight_layout()
    fig.savefig(FIGURES / "acc_by_type_test.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(labels) + 1.6))
    hbars(ax, labels, [("note count right", pct(cnt), C3), ("notes on the right step", pct(notes), C2),
                       ("neume entirely right", pct(whole), C1)],
          counts, (0, 106), "Test set, final model: miscounted notes, or misplaced ones?")
    fig.tight_layout()
    fig.savefig(FIGURES / "acc_by_type_errors.png", dpi=130)
    plt.close(fig)

    ds, s1_salz = salzinnes(dev)
    ts = types(ds)
    _, _, w_before = scores(ds, *predict(BEFORE, ds, dev))
    c_after, _, w_after = scores(ds, *predict(FINAL, ds, dev))
    ts = pool(ts, ds["n_notes"], 10)
    labels_s, counts_s = order(ts, ds["n_notes"], 1)
    pcts = lambda v: [100 * float(np.mean(v[ts == l])) for l in labels_s]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(labels_s) + 1.6))
    hbars(ax, labels_s, [("without spacing jitter", pcts(w_before), C3), ("final model", pcts(w_after), C1),
                         ("one crop per note (step 1)", pcts(s1_salz), "#bfc3c9")],
          counts_s, (0, 106), "Salzinnes (unseen manuscript): neumes entirely right, by type")
    fig.tight_layout()
    fig.savefig(FIGURES / "acc_by_type_salzinnes.png", dpi=130)
    plt.close(fig)

    sizes = ["1", "2", "3", "4", "5+"]
    bucket = lambda n: np.where(n >= 5, "5+", n.astype(str))
    bt, bs = bucket(d["n_notes"]), bucket(ds["n_notes"])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(sizes))
    for k, (name, vals, col) in enumerate([
            ("test, final model", [100 * whole[bt == s].mean() for s in sizes], C1),
            ("Salzinnes, without spacing jitter", [100 * w_before[bs == s].mean() for s in sizes], C3),
            ("Salzinnes, final model", [100 * w_after[bs == s].mean() for s in sizes], C2)]):
        ax.bar(x + (k - 1) * 0.27, vals, width=0.25, color=col, label=name)
        for xi, v in zip(x + (k - 1) * 0.27, vals):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, [f"{s} notes\ntest n={int((bt == s).sum()):,}\nSalz. n={int((bs == s).sum())}" for s in sizes])
    ax.set_ylim(0, 108)
    ax.set_ylabel("neumes entirely right (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.set_title("Neumes entirely right, by number of notes", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "acc_by_size.png", dpi=130)
    plt.close(fig)
    for f in ("acc_by_type_test", "acc_by_type_errors", "acc_by_type_salzinnes", "acc_by_size"):
        print(f"wrote figures/{f}.png")


if __name__ == "__main__":
    main()

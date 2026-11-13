"""Step 1: v2's staff-free model on every note, each read from its own MEI box.

v2's models were trained on single-note neumes only. Here v2's staff-free
configuration (cls_aug_bbox128*: 128 x 32, the note's box +-10 units, no staff)
is trained on every usable note, multi-note neumes included, each note cut from
its own MEI box exactly as a single note is (v2's build_crops.py --mode bbox),
with v2's training loop. Both of v2's box jitters: two notes stacked in one
column (a podatus) differ only in their box's height on the page, which the
final model's +-1.5-step jitter teaches it to distrust.

Needs v2's bbox128 cache with every note -- a superset of the single-note one,
so v2's models are unaffected:
    cd ../v2-square/dataset && python build_crops.py --mode bbox --img-h 128

    python -u train_notes.py 2>&1 | tee checkpoints/train_notes.log
"""
import argparse
import json

import numpy as np
import torch

from paths import CKPT, DATA
import train as v2  # v2's train.py
from data import load  # v2's data.py

CONFIGS = {
    "cls_aug_bbox128_all": v2._cfg(True, True, DATA / "v2_crops_bbox128", 128, 8, v2.BBOX_SHIFT),
    "cls_aug_bbox128_shift15_all": v2._cfg(True, True, DATA / "v2_crops_bbox128", 128, 8, 1.5 / 20),
}


def load_model(name, dev):
    """(model, config) for a note model: one of these (v3's checkpoints) or any of v2's."""
    cfg, ckpt = (CONFIGS[name], CKPT) if name in CONFIGS else (v2.CONFIGS[name], v2.CKPT)
    m = v2.build_model(name, cfg).to(dev)
    m.load_state_dict(torch.load(ckpt / f"{name}.pt", map_location=dev))
    return m.eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(CONFIGS), default=list(CONFIGS))
    args = ap.parse_args()
    CKPT.mkdir(exist_ok=True)
    dev = v2.device()
    meta_path = CKPT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for name in args.only:
        c = CONFIGS[name]
        tr = load(v2.MANUSCRIPTS, "train", single_note_only=False, crops_dir=c["crops"])
        va = load(v2.MANUSCRIPTS, "val", single_note_only=False, crops_dir=c["crops"])
        counts = {s: {ms: int(np.sum(d["manuscript"] == ms)) for ms in v2.MANUSCRIPTS} for s, d in (("train", tr), ("val", va))}
        print(f"training {name} on {c['crops'].name} (device {dev}; all crops {counts})", flush=True)
        meta[name] = v2.train_one(name, tr, va, dev, cfg=c, ckpt_dir=CKPT)
        meta[name].update(seed=v2.SEED, epochs=v2.EPOCHS, batch=v2.BATCH, lr=v2.LR, weight_decay=v2.WD,
                          n_train=counts["train"], crops=c["crops"].name, out_h=c["out_h"], pool=c["pool"],
                          shift_jitter=c["shift"], mask=c["mask"], all_notes=True)
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"  {name}: best val exact {100 * meta[name]['best_val_exact']:.2f}% at epoch {meta[name]['best_epoch']}",
              flush=True)


if __name__ == "__main__":
    main()

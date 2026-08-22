"""Train the v1 StepCNN and save checkpoints for evaluate.py to reuse.

    python train.py
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import StepCNN
from data import REAL_LABELED_PAGES, PSEUDO_ELIGIBLE_PAGES, load_real_labeled_page, load_pseudo_page, load_shapes
from split import build_split, SEED

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "checkpoints"
EPOCHS = 300
LR = 1e-3
BATCH = 16


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_cnn(Xtr, ytr, seed=SEED):
    set_seed(seed)
    model = StepCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
    n = len(Xtr_t)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
    return model


def main():
    set_seed(SEED)
    shapes = load_shapes()

    print("loading real-labeled pages...")
    all_rows = []
    for page in REAL_LABELED_PAGES:
        rows = load_real_labeled_page(page, shapes)
        print(f"  {page}: {len(rows)}")
        all_rows += rows

    print("building split...")
    train_rows, test_rows = build_split(all_rows)
    print(f"train={len(train_rows)} test={len(test_rows)}")

    X_train = np.stack([r["crop"] for r in train_rows])[:, None, :, :]
    y_train = np.array([r["truth"] for r in train_rows], dtype=np.float32)

    CKPT_DIR.mkdir(exist_ok=True)

    print("training (real-only)...")
    model_real = train_cnn(X_train, y_train)
    torch.save(model_real.state_dict(), CKPT_DIR / "stepcnn_real.pt")

    print("loading pseudo-label pool...")
    pseudo_rows = []
    for page in PSEUDO_ELIGIBLE_PAGES:
        rows = load_pseudo_page(page, shapes)
        print(f"  {page}: {len(rows)} pseudo")
        pseudo_rows += rows

    if pseudo_rows:
        X_pseudo = np.stack([r["crop"] for r in pseudo_rows])[:, None, :, :]
        y_pseudo = np.array([r["step"] for r in pseudo_rows], dtype=np.float32)
        X_train_p = np.concatenate([X_train, X_pseudo], axis=0)
        y_train_p = np.concatenate([y_train, y_pseudo], axis=0)
    else:
        X_train_p, y_train_p = X_train, y_train

    print(f"training (real+pseudo, n_pseudo={len(pseudo_rows)})...")
    model_pseudo = train_cnn(X_train_p, y_train_p)
    torch.save(model_pseudo.state_dict(), CKPT_DIR / "stepcnn_real_pseudo.pt")

    meta = {
        "seed": SEED, "n_train_real": len(train_rows), "n_test": len(test_rows),
        "n_pseudo": len(pseudo_rows), "pages": REAL_LABELED_PAGES,
    }
    (CKPT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved checkpoints to {CKPT_DIR}")


if __name__ == "__main__":
    main()

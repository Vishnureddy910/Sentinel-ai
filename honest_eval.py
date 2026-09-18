"""How often is the model actually right, across ALL held-out images?"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.dataset import DISEASES, ChestXrayDataset, build_transforms
from models.resnet_model import get_model


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = get_model()
    m.load_state_dict(torch.load("data/models/global_best.pth", map_location="cpu"))
    m.to(dev).eval()
    calib = json.loads(Path("data/models/thresholds.json").read_text())
    th = np.array([calib["thresholds"][d] for d in DISEASES])

    d = Path("data/client_1_prep")
    ds = ChestXrayDataset(d / "val.csv", d, build_transforms(train=False))
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)

    P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                P.append(torch.sigmoid(m(x.to(dev)).float()).cpu().numpy())
            Y.append(y.numpy())
    P, Y = np.vstack(P), np.vstack(Y)
    print(f"Evaluated {len(P)} held-out images\n")

    pred = P >= th
    print(f"{'Disease':20s} {'AUC':>6s} {'recall':>7s} {'precis':>7s} {'top1hit':>8s} {'n':>6s}")
    print("-" * 60)
    for i, dis in enumerate(DISEASES):
        y, p = Y[:, i], pred[:, i]
        n = int(y.sum())
        if n == 0:
            continue
        recall = p[y == 1].mean()
        prec = y[p].mean() if p.sum() else 0.0
        # of images where this disease IS present, how often is it the top-1 score?
        top1 = (P[y == 1].argmax(axis=1) == i).mean()
        print(f"{dis:20s} {calib['auc'][dis]:6.3f} {recall:7.1%} {prec:7.1%} {top1:8.1%} {n:6d}")

    print("-" * 60)
    any_true = Y.sum(axis=1) > 0
    # "is the model's single top guess one of the true labels?"
    top1_correct = np.array([Y[k, P[k].argmax()] == 1 for k in range(len(P))])
    print(f"Images with >=1 finding: {any_true.sum()} / {len(P)}")
    print(f"Top-1 guess is a TRUE label: {top1_correct[any_true].mean():.1%}")
    healthy = ~any_true
    print(f"Healthy images flagged as clean: {(pred[healthy].sum(axis=1) == 0).mean():.1%}")
    print(f"Avg diseases flagged per image: {pred.sum(axis=1).mean():.1f} (true avg: {Y.sum(axis=1).mean():.1f})")


if __name__ == "__main__":
    main()

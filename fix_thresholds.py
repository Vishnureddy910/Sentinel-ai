"""Re-calibrate decision thresholds for F1 instead of Youden's J.

Youden's J maximises TPR-FPR, which ignores class imbalance: at 0.19% prevalence
a threshold with 5% FPR produces ~25 false positives per true positive. F1
balances precision against recall, which is what a demo actually needs.

Calibrated on hospital 2's validation patients, reported on hospital 1's, so the
numbers are not tuned and measured on the same images.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.dataset import DISEASES, ChestXrayDataset, build_transforms
from models.resnet_model import get_model


def score(model, d, dev):
    ds = ChestXrayDataset(d / "val.csv", d, build_transforms(train=False))
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)
    P, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                P.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
            Y.append(y.numpy())
    return np.vstack(P), np.vstack(Y)


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = get_model()
    m.load_state_dict(torch.load("data/models/global_best.pth", map_location="cpu"))
    m.to(dev).eval()

    Pc, Yc = score(m, Path("data/client_2_prep"), dev)   # calibrate here
    Pr, Yr = score(m, Path("data/client_1_prep"), dev)   # report here
    print(f"calibrate on {len(Pc)} images, report on {len(Pr)}\n")

    old = json.loads(Path("data/models/thresholds.json").read_text())
    new_th = {}
    grid = np.linspace(0.05, 0.99, 190)
    for i, dis in enumerate(DISEASES):
        y, p = Yc[:, i], Pc[:, i]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = p >= t
            tp = (pred & (y == 1)).sum()
            if tp == 0:
                continue
            f1 = 2 * tp / (pred.sum() + y.sum())
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        new_th[dis] = best_t

    print(f"{'Disease':20s} {'OLD thr':>8s} {'NEW thr':>8s} | {'OLD prec':>8s} {'NEW prec':>8s} | {'OLD rec':>8s} {'NEW rec':>8s}")
    print("-" * 82)
    for i, dis in enumerate(DISEASES):
        y = Yr[:, i]
        if y.sum() == 0:
            continue
        stats = []
        for t in (old["thresholds"][dis], new_th[dis]):
            pred = Pr[:, i] >= t
            prec = y[pred].mean() if pred.sum() else 0.0
            stats += [prec, pred[y == 1].mean()]
        print(f"{dis:20s} {old['thresholds'][dis]:8.3f} {new_th[dis]:8.3f} | "
              f"{stats[0]:7.1%} {stats[2]:8.1%} | {stats[1]:7.1%} {stats[3]:8.1%}")

    for tag, th in (("OLD (Youden J)", np.array([old["thresholds"][d] for d in DISEASES])),
                    ("NEW (F1)", np.array([new_th[d] for d in DISEASES]))):
        pred = Pr >= th
        healthy = Yr.sum(axis=1) == 0
        print(f"\n{tag}: avg flags/image {pred.sum(axis=1).mean():.1f} (true 0.7) | "
              f"healthy correctly clean {(pred[healthy].sum(axis=1)==0).mean():.1%}")

    out = dict(old)
    out["thresholds"] = new_th
    out["calibration"] = "F1-optimal, fitted on hospital 2 val"
    Path("data/models/thresholds.json").write_text(json.dumps(out, indent=2))
    print("\nWrote updated data/models/thresholds.json")


if __name__ == "__main__":
    main()

"""
Score the federated global model on the held-out validation patients of every
hospital, and calibrate a decision threshold per disease.

Why thresholds matter: training uses BCE with per-class pos_weight to fight the
class imbalance, which deliberately pushes the sigmoid outputs upward. A flat
"probability > 0.10 means positive" rule would then flag almost everything. We
pick each class's threshold on validation data via Youden's J (max TPR - FPR).

    python evaluate_global.py [path/to/checkpoint.pth]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import ConcatDataset, DataLoader

from models.dataset import DISEASES, ChestXrayDataset, build_transforms
from models.resnet_model import get_model

REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "data" / "models"


def main():
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_DIR / "global_best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt}. Train first, or pass a path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device).eval()
    print(f"Loaded {ckpt} on {device}")

    val_sets = []
    for client_dir in sorted(REPO_ROOT.glob("data/client_*_prep")):
        val_csv = client_dir / "val.csv"
        if val_csv.exists():
            val_sets.append(ChestXrayDataset(val_csv, client_dir, build_transforms(train=False)))
            print(f"  + {client_dir.name}/val.csv ({len(val_sets[-1])} images)")
    if not val_sets:
        raise FileNotFoundError("No prepared validation sets found. Run prepare_data.py first.")

    loader = DataLoader(ConcatDataset(val_sets), batch_size=128, shuffle=False,
                        num_workers=12, pin_memory=True)

    probs, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(images)
            probs.append(torch.sigmoid(out.float()).cpu().numpy())
            targets.append(labels.numpy())
    probs = np.vstack(probs)
    targets = np.vstack(targets)
    print(f"\nScored {len(probs)} held-out images\n")

    thresholds, aucs = {}, {}
    print(f"{'Disease':22s} {'AUC':>7s} {'thresh':>8s} {'prev':>7s}")
    print("-" * 47)
    for i, disease in enumerate(DISEASES):
        y, p = targets[:, i], probs[:, i]
        if y.min() == y.max():
            print(f"{disease:22s} {'n/a':>7s}")
            continue
        auc = roc_auc_score(y, p)
        fpr, tpr, cuts = roc_curve(y, p)
        best = cuts[int(np.argmax(tpr - fpr))]
        aucs[disease] = float(auc)
        thresholds[disease] = float(best)
        print(f"{disease:22s} {auc:7.3f} {best:8.3f} {y.mean()*100:6.2f}%")

    mean_auc = float(np.mean(list(aucs.values())))
    print("-" * 47)
    print(f"{'MEAN AUC':22s} {mean_auc:7.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "thresholds.json"
    out.write_text(json.dumps({"mean_auc": mean_auc, "auc": aucs, "thresholds": thresholds}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

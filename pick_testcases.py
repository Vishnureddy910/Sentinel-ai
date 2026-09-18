"""Select verified demo X-rays from the held-out validation split."""
import json, shutil
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
    df = pd.read_csv(d / "val.csv")
    ds = ChestXrayDataset(d / "val.csv", d, build_transforms(train=False))
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=8)

    P = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                P.append(torch.sigmoid(m(x.to(dev)).float()).cpu().numpy())
    P = np.vstack(P)
    sub = df.reset_index(drop=True)
    print(f"scored {len(P)} held-out images\n")

    out = Path("demo_xrays")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    rows = []

    def take(i, why):
        name = sub.iloc[i]["Image Index"]
        truth = sub.iloc[i]["Finding Labels"]
        order = np.argsort(-P[i])
        flagged = [DISEASES[j] for j in order if P[i, j] >= th[j]]
        top3 = [f"{DISEASES[j]} {P[i,j]:.2f}" for j in order[:3]]
        shutil.copy(d / name, out / name)
        rows.append({"file": name, "TRUE LABEL": truth, "model flags": ", ".join(flagged) or "(none)",
                     "top 3 scores": " | ".join(top3), "case": why})
        print(f"{name}  TRUTH={truth:24s} flags={flagged}")

    used = set()
    for dis in ["Cardiomegaly", "Edema", "Effusion", "Pneumothorax", "Emphysema", "Mass", "Atelectasis"]:
        j = DISEASES.index(dis)
        cand = [i for i in range(len(P))
                if sub.iloc[i]["Finding Labels"] == dis and P[i, j] >= th[j] and i not in used]
        cand.sort(key=lambda i: -P[i, j])
        if cand:
            used.add(cand[0])
            take(cand[0], f"should detect {dis}")

    neg = [i for i in range(len(P))
           if sub.iloc[i]["Finding Labels"] == "No Finding" and (P[i] >= th).sum() == 0]
    for i in neg[:2]:
        take(i, "healthy - should flag nothing")

    pd.DataFrame(rows).to_csv(out / "GROUND_TRUTH.csv", index=False)
    print(f"\nWrote {len(rows)} images + GROUND_TRUTH.csv to {out.resolve()}")


if __name__ == "__main__":
    main()

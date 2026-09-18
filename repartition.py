"""Re-partition the prepared images into N hospitals for the Byzantine experiment.

Krum's robustness guarantee needs n > 2f + 2, so tolerating one attacker needs
at least 5 clients. This reuses the already-resized 256px images (hardlinked, so
no extra disk) and only rebuilds the patient-wise CSV splits.
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
IMAGES = DATA / "prep_images"
SPLITS = DATA / "splits"
VAL_FRACTION = 0.15
SEED = 42


def main():
    n_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    IMAGES.mkdir(parents=True, exist_ok=True)
    frames = []
    linked = 0
    for prep in sorted(DATA.glob("client_*_prep")):
        for split in ("train", "val"):
            csv = prep / f"{split}.csv"
            if csv.exists():
                frames.append(pd.read_csv(csv))
        for png in prep.glob("*.png"):
            dst = IMAGES / png.name
            if not dst.exists():
                try:
                    os.link(png, dst)          # hardlink: same bytes, no extra space
                except OSError:
                    shutil.copy(png, dst)
                linked += 1
    print(f"linked {linked} images into {IMAGES}")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Image Index"])
    df = df[df["Image Index"].isin({p.name for p in IMAGES.glob("*.png")})]
    print(f"{len(df)} images, {df['Patient ID'].nunique()} patients")

    patients = np.array(sorted(df["Patient ID"].unique()))
    np.random.default_rng(SEED).shuffle(patients)
    shards = np.array_split(patients, n_clients)

    if SPLITS.exists():
        shutil.rmtree(SPLITS)
    for cid, shard in enumerate(shards, start=1):
        out = SPLITS / f"client_{cid}"
        out.mkdir(parents=True)
        sub = df[df["Patient ID"].isin(set(shard))].copy()

        shuffled = shard.copy()
        np.random.default_rng(SEED + cid).shuffle(shuffled)
        val_patients = set(shuffled[: max(1, int(len(shuffled) * VAL_FRACTION))])
        is_val = sub["Patient ID"].isin(val_patients)

        sub[~is_val].to_csv(out / "train.csv", index=False)
        sub[is_val].to_csv(out / "val.csv", index=False)
        print(f"  hospital {cid}: {len(shard)} patients, "
              f"{(~is_val).sum()} train / {is_val.sum()} val")

    print(f"\nDone -> {SPLITS}")


if __name__ == "__main__":
    main()

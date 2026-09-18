"""
One-time data preparation for Sentinel-AI.

Fixes three problems with the original alternating-image split:
  1. Images were 1024x1024 PNGs (~400 KB) decoded from scratch every round.
     We pre-resize once to 256x256 (~25 KB), a ~25x decode speedup per worker.
  2. The same patient's scans landed in BOTH hospitals (13,302 patients
     overlapped). We partition by Patient ID so the hospitals are disjoint.
  3. There was no validation set at all. We carve a patient-disjoint
     validation split inside each hospital, so reported AUC is honest.

Run once:  python prepare_data.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MASTER_CSV = DATA_DIR / "data" / "Data_Entry_2017.csv"

# Store at 256 so training can RandomCrop to 224 (augmentation needs the margin)
STORE_SIZE = 256
VAL_FRACTION = 0.15
SEED = 42
NUM_CLIENTS = 2

DISEASES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def index_source_images():
    """Map image filename -> current location on disk, wherever it ended up."""
    index = {}
    for png in DATA_DIR.rglob("*.png"):
        # Never treat our own generated 256px copies as a source
        if "_prep" in png.parent.name:
            continue
        index.setdefault(png.name, png)
    return index


def _resize_one(args):
    src, dst = args
    try:
        with Image.open(src) as im:
            im.resize((STORE_SIZE, STORE_SIZE), Image.BILINEAR).save(dst, optimize=False, compress_level=1)
        return True
    except Exception as exc:  # corrupt/truncated file -> drop the row later
        print(f"  ! failed {src}: {exc}", file=sys.stderr)
        return False


def main():
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Master CSV not found at {MASTER_CSV}")

    print("Indexing source images...")
    index = index_source_images()
    print(f"  found {len(index)} PNGs on disk")

    df = pd.read_csv(MASTER_CSV)
    df = df[df["Image Index"].isin(index.keys())].copy()
    print(f"  {len(df)} CSV rows have a matching image")

    # ---- Patient-wise partition across hospitals (no patient in two hospitals) ----
    patients = np.array(sorted(df["Patient ID"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)
    shards = np.array_split(patients, NUM_CLIENTS)

    for client_id, shard in enumerate(shards, start=1):
        out_dir = DATA_DIR / f"client_{client_id}_prep"
        out_dir.mkdir(parents=True, exist_ok=True)

        client_df = df[df["Patient ID"].isin(set(shard))].copy()

        # ---- Patient-wise train/val split inside this hospital ----
        shard_shuffled = shard.copy()
        np.random.default_rng(SEED + client_id).shuffle(shard_shuffled)
        n_val = max(1, int(len(shard_shuffled) * VAL_FRACTION))
        val_patients = set(shard_shuffled[:n_val])
        client_df["split"] = np.where(client_df["Patient ID"].isin(val_patients), "val", "train")

        print(f"\nHospital {client_id}: {len(client_df)} images, {len(shard)} patients "
              f"({len(client_df[client_df.split=='train'])} train / {len(client_df[client_df.split=='val'])} val)")

        # ---- Parallel resize ----
        jobs = [(str(index[name]), str(out_dir / name))
                for name in client_df["Image Index"]
                if not (out_dir / name).exists()]
        if jobs:
            print(f"  resizing {len(jobs)} images to {STORE_SIZE}x{STORE_SIZE}...")
            workers = min(32, (os.cpu_count() or 4))
            done = 0
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for _ in pool.map(_resize_one, jobs, chunksize=64):
                    done += 1
                    if done % 5000 == 0:
                        print(f"    {done}/{len(jobs)}")
        else:
            print("  images already prepared, skipping resize")

        # Drop rows whose image failed to write
        present = {p.name for p in out_dir.glob("*.png")}
        client_df = client_df[client_df["Image Index"].isin(present)]

        for split in ("train", "val"):
            split_df = client_df[client_df["split"] == split]
            split_df.to_csv(out_dir / f"{split}.csv", index=False)
            print(f"  wrote {split}.csv ({len(split_df)} rows)")

    print("\nDone. Point the clients at data/client_N_prep/.")


if __name__ == "__main__":
    main()

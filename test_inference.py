import json
import sys
from pathlib import Path

import torch
from PIL import Image

from models.dataset import DISEASES, inference_transform
from models.resnet_model import get_model

# Windows cmd defaults to cp1252 and dies on the emoji below; force UTF-8 and
# degrade gracefully on consoles that still cannot render it.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = DISEASES  # single source of truth, so the order can never drift


def find_checkpoint():
    """Prefer the best-validation-AUC model the federated server saved."""
    for c in [ROOT / "data" / "models" / "global_best.pth",
              ROOT / "data" / "models" / "global_latest.pth"]:
        if c.exists():
            return c
    legacy = sorted((ROOT / "data").glob("global_model_round_*.pth"))
    return legacy[-1] if legacy else None


def load_thresholds():
    """Per-class cut-offs calibrated on held-out patients (see fix_thresholds.py).
    Training uses pos_weight to fight class imbalance, which inflates the raw
    sigmoid scores -- a flat 0.10 rule would flag nearly every disease."""
    path = ROOT / "data" / "models" / "thresholds.json"
    if path.exists():
        calib = json.loads(path.read_text())
        return calib.get("thresholds", {}), calib.get("auc", {}), calib.get("mean_auc")
    return {}, {}, None


def load_truth(test_dir):
    """If the folder carries a GROUND_TRUTH.csv, show the real labels too."""
    csv = test_dir / "GROUND_TRUTH.csv"
    if not csv.exists():
        return {}
    import csv as _csv
    with csv.open(newline="") as fh:
        return {r["file"]: r["TRUE LABEL"] for r in _csv.DictReader(fh)}


ckpt = find_checkpoint()
if ckpt is None:
    sys.exit("No checkpoint found. Train first, or copy data/models/global_best.pth into place.")

print("Loading Sentinel-AI Global Model...")
model = get_model(pretrained=False)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

cuts, class_auc, mean_auc = load_thresholds()
print(f"Model: {ckpt.name}" + (f" | validation mean AUC {mean_auc:.3f}" if mean_auc else ""))
if not cuts:
    print("WARNING: no thresholds.json -- falling back to 0.5. Run fix_thresholds.py.")

# Default to the verified demo X-rays; accept a folder as an argument.
test_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
    (d for d in [ROOT / "demo_xrays", ROOT / "data" / "sample_test"] if d.exists()),
    ROOT / "demo_xrays")
truth = load_truth(test_dir)

image_paths = sorted(p for p in test_dir.iterdir()
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg")) if test_dir.exists() else []

if not image_paths:
    print(f"No images found in {test_dir}. Please add some scans!")
    sys.exit(0)

print(f"Found {len(image_paths)} images to analyze. Starting batch inference...\n")

for img_path in image_paths:
    print("========================================")
    print(f"🩺 Patient Scan: {img_path.name}")
    if img_path.name in truth:
        print(f"   Ground truth: {truth[img_path.name]}")
    print("========================================")

    image = Image.open(img_path).convert("RGB")
    # Same preprocessing the model was validated with (256 -> centre 224).
    tensor = inference_transform()(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).squeeze().tolist()

    predictions = sorted(zip(CLASS_NAMES, probs), key=lambda x: x[1], reverse=True)

    flagged = False
    for disease, prob in predictions:
        cut = cuts.get(disease, 0.5)
        if prob >= cut:
            flagged = True
            auc_note = f"  (class AUC {class_auc[disease]:.2f})" if disease in class_auc else ""
            print(f"[DETECTED] {disease:20s}: {prob*100:6.2f}%  threshold {cut*100:5.1f}%{auc_note}")

    if not flagged:
        top, p = predictions[0]
        print(f"[CLEAR] No pathology above its threshold (highest: {top} {p*100:.1f}%)")
    print()

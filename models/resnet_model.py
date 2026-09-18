import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

from models.dataset import DISEASES, ChestXrayDataset, build_transforms

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_weights():
    """Find the ImageNet checkpoint whether we're in Docker or on the host."""
    candidates = [
        os.environ.get("SENTINEL_RESNET_WEIGHTS"),
        _REPO_ROOT / "resnet18.pth",
        Path("/app/resnet18.pth"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def get_model(pretrained=True):
    """ResNet-18 backbone with the classifier head resized to 14 diseases.

    pretrained=False skips the ImageNet initialisation entirely. Callers that
    immediately load a trained checkpoint should pass False: the ImageNet weights
    would be overwritten anyway, and skipping them means inference needs no
    network access and no local resnet18.pth.
    """
    model = models.resnet18()
    if pretrained:
        weights_path = _resolve_weights()
        if weights_path is not None:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        else:
            # Fall back to downloading rather than silently training from
            # scratch, which would cost a large amount of AUC.
            print("WARNING: local resnet18.pth not found, downloading ImageNet weights")
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(DISEASES))
    return model


def _macro_auc(labels, preds):
    """Mean AUC over classes that actually have both positives and negatives.

    A successfully poisoned global model diverges and emits NaN logits. That is a
    model with no ranking ability whatsoever, which is chance level by
    definition, so we score it 0.5 rather than crashing the federated round.
    """
    finite = np.isfinite(preds)
    if not finite.all():
        print(f"WARNING: {(~finite).mean():.1%} of predictions are NaN/inf "
              f"(model has diverged); scoring those at chance", flush=True)
        preds = np.where(finite, preds, 0.5)

    aucs = {}
    for i, disease in enumerate(DISEASES):
        col = labels[:, i]
        if col.min() == col.max():
            continue  # undefined for this batch/split
        aucs[disease] = roc_auc_score(col, preds[:, i])
    mean = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return mean, aucs


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, max_batches=None):
    model.train()
    running_loss = 0.0
    n_batches = 0
    all_labels, all_preds = [], []

    use_amp = scaler is not None and device.type == "cuda"

    for images, labels in dataloader:
        # Each round consumes a random slice of the local data rather than a full
        # epoch -- shuffle=True means the first max_batches are a random subset.
        if max_batches is not None and n_batches >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        all_labels.append(labels.detach().cpu().numpy())
        # Logits -> probabilities for the AUC metric (BCEWithLogitsLoss applies
        # the sigmoid internally, so it never leaves the loss path).
        all_preds.append(torch.sigmoid(outputs.detach().float()).cpu().numpy())

    epoch_loss = running_loss / max(n_batches, 1)
    mean_auc, _ = _macro_auc(np.vstack(all_labels), np.vstack(all_preds))
    return epoch_loss, mean_auc


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device, use_amp=True):
    """Honest validation pass on held-out patients. Returns (loss, mean AUC, per-class AUC)."""
    model.eval()
    running_loss = 0.0
    n_batches = 0
    all_labels, all_preds = [], []

    amp_on = use_amp and device.type == "cuda"

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_on):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        n_batches += 1
        all_labels.append(labels.cpu().numpy())
        all_preds.append(torch.sigmoid(outputs.float()).cpu().numpy())

    val_loss = running_loss / max(n_batches, 1)
    mean_auc, per_class = _macro_auc(np.vstack(all_labels), np.vstack(all_preds))
    return val_loss, mean_auc, per_class


if __name__ == "__main__":
    print("--- Centralized Baseline Training ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = _REPO_ROOT / "data" / "client_1_prep"
    train_ds = ChestXrayDataset(data_dir / "train.csv", data_dir, build_transforms(train=True))
    val_ds = ChestXrayDataset(data_dir / "val.csv", data_dir, build_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_ds.pos_weight().to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(3):
        loss, auc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_auc, per_class = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} - train loss {loss:.4f} auc {auc:.4f} | val loss {val_loss:.4f} AUC {val_auc:.4f}")

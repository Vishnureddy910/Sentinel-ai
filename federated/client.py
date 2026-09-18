import math
import os
import sys
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.dataset import DISEASES, ChestXrayDataset, build_transforms
from models.resnet_model import evaluate_model, get_model, train_one_epoch

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tunables (overridable per-container from docker-compose)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", 128))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 16))
SAMPLES_PER_ROUND = int(os.environ.get("SAMPLES_PER_ROUND", 12000))
DEFAULT_LR = float(os.environ.get("LEARNING_RATE", 1e-4))

# Byzantine behaviour for the robustness experiment:
#   none      - honest hospital
#   signflip  - model poisoning: return the negated, scaled update
#   labelflip - data poisoning: train on inverted labels
ATTACK = os.environ.get("ATTACK", "none").lower()
ATTACK_SCALE = float(os.environ.get("ATTACK_SCALE", 3.0))


class FlippedLabelLoss(nn.Module):
    """Trains the model towards the exact opposite of the truth."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, outputs, labels):
        return self.base(outputs, 1.0 - labels)


class SentinelClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, val_loader, pos_weight, device):
        self.client_id = client_id
        self.model = get_model().to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Per-class weighting from this hospital's own label distribution. A flat
        # weight would swamp common findings and ignore rare ones like Hernia.
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        if ATTACK == "labelflip":
            self.criterion = FlippedLabelLoss(self.criterion)

        # Optimizer lives across rounds so Adam's moment estimates survive; a
        # fresh optimizer each round throws away that adaptation.
        self.optimizer = optim.Adam(self.model.parameters(), lr=DEFAULT_LR)
        self.scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

        self.max_batches = max(1, math.ceil(SAMPLES_PER_ROUND / BATCH_SIZE))

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v, dtype=self.model.state_dict()[k].dtype) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr = float(config.get("lr", DEFAULT_LR))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        loss, auc = train_one_epoch(
            self.model, self.train_loader, self.criterion, self.optimizer,
            self.device, self.scaler, max_batches=self.max_batches,
        )
        n_seen = self.max_batches * BATCH_SIZE
        rnd = config.get("server_round", "?")

        params = self.get_parameters(config={})
        if ATTACK == "signflip":
            # Push the global average in the opposite direction, scaled up so a
            # single attacker can drag the mean with it.
            params = [(-ATTACK_SCALE * p).astype(p.dtype) for p in params]

        tag = "" if ATTACK == "none" else f" [MALICIOUS:{ATTACK}]"
        print(f"[client {self.client_id}]{tag} round {rnd} lr={lr:.2e} "
              f"train loss {loss:.4f} auc {auc:.4f}", flush=True)

        return params, n_seen, {"train_loss": float(loss), "train_auc": float(auc)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        val_loss, val_auc, per_class = evaluate_model(
            self.model, self.val_loader, self.criterion, self.device
        )
        rnd = config.get("server_round", "?")
        print(f"[client {self.client_id}] round {rnd} VAL loss {val_loss:.4f} mean AUC {val_auc:.4f}", flush=True)
        for disease, score in sorted(per_class.items(), key=lambda kv: -kv[1]):
            print(f"    {disease:20s} {score:.3f}", flush=True)

        metrics = {"val_auc": float(val_auc)}
        metrics.update({f"auc_{d}": float(s) for d, s in per_class.items()})
        return float(val_loss), len(self.val_loader.dataset), metrics


def build_loaders(client_id):
    # Preferred: N-hospital layout from repartition.py (CSVs + one shared image dir)
    split_dir = REPO_ROOT / "data" / "splits" / f"client_{client_id}"
    if (split_dir / "train.csv").exists():
        csv_dir, img_dir = split_dir, REPO_ROOT / "data" / "prep_images"
    else:
        # Fallback: original 2-hospital layout from prepare_data.py
        csv_dir = img_dir = REPO_ROOT / "data" / f"client_{client_id}_prep"
    if not (csv_dir / "train.csv").exists():
        raise FileNotFoundError(
            f"No split for client {client_id}. Run `python prepare_data.py`, "
            f"then `python repartition.py 5` for the multi-client experiment."
        )

    train_ds = ChestXrayDataset(csv_dir / "train.csv", img_dir, build_transforms(train=True))
    val_ds = ChestXrayDataset(csv_dir / "val.csv", img_dir, build_transforms(train=False))

    # num_workers is the single biggest lever here: with the default of 0 the
    # GPU sits idle ~97% of the time waiting on PNG decodes.
    common = dict(num_workers=NUM_WORKERS, pin_memory=True)
    if NUM_WORKERS > 0:
        common.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, **common)
    return train_ds, train_loader, val_loader


if __name__ == "__main__":
    client_id = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CLIENT_ID", "1")
    server_ip = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("SERVER_ADDRESS", "127.0.0.1:8080")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    role = "HONEST" if ATTACK == "none" else f"MALICIOUS ({ATTACK})"
    print(f"--- Hospital Node {client_id} | {role} | device={device} | workers={NUM_WORKERS} ---", flush=True)

    train_ds, train_loader, val_loader = build_loaders(client_id)
    pos_weight = train_ds.pos_weight()
    print(f"train={len(train_ds)} val={len(val_loader.dataset)} "
          f"samples/round={SAMPLES_PER_ROUND}", flush=True)

    client = SentinelClient(client_id, train_loader, val_loader, pos_weight, device)
    fl.client.start_client(server_address=server_ip, client=client.to_client())

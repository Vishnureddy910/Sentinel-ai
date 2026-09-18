import math
import os
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import torch

from models.resnet_model import get_model

NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", 40))
INITIAL_LR = float(os.environ.get("LEARNING_RATE", 1e-4))
MIN_LR = float(os.environ.get("MIN_LR", 1e-5))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", 2))
NUM_MALICIOUS = int(os.environ.get("NUM_MALICIOUS", 0))

# "krum" = Multi-Krum (Byzantine-robust), "fedavg" = plain averaging (no defense)
STRATEGY = os.environ.get("STRATEGY", "krum").lower()
EXPERIMENT = os.environ.get("EXPERIMENT", STRATEGY)

SAVE_DIR = Path(os.environ.get("MODEL_DIR", "/app/data/models"))


def fit_config(server_round: int):
    """Cosine-decay the learning rate across rounds -- large steps early to adapt
    the ImageNet features, small steps late so the federated average settles."""
    progress = (server_round - 1) / max(NUM_ROUNDS - 1, 1)
    lr = MIN_LR + 0.5 * (INITIAL_LR - MIN_LR) * (1 + math.cos(math.pi * progress))
    return {"server_round": server_round, "lr": lr}


def evaluate_config(server_round: int):
    return {"server_round": server_round}


def weighted_average(metrics):
    """Aggregate client metrics weighted by each client's number of examples."""
    if not metrics:
        return {}
    total = sum(n for n, _ in metrics)
    keys = set()
    for _, m in metrics:
        keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
    return {k: sum(n * m[k] for n, m in metrics if k in m) / total for k in keys}


def make_saving_strategy(base_cls):
    """Wrap any Flower strategy with checkpointing and per-round AUC logging, so
    Multi-Krum and FedAvg can be compared under identical conditions."""

    class SaveModel(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.best_auc = float("-inf")
            self._latest_state = None
            SAVE_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = SAVE_DIR / f"history_{EXPERIMENT}.csv"
            self.log_path.write_text("round,val_auc,val_loss\n")

        def _to_state_dict(self, parameters):
            ndarrays = fl.common.parameters_to_ndarrays(parameters)
            reference = get_model(pretrained=False).state_dict()
            # Cast back to each tensor's original dtype -- averaging promotes the
            # integer BatchNorm counters to float and load_state_dict is strict.
            return OrderedDict(
                {k: torch.tensor(v, dtype=reference[k].dtype)
                 for k, v in zip(reference.keys(), ndarrays)}
            )

        def aggregate_fit(self, server_round, results, failures):
            params, metrics = super().aggregate_fit(server_round, results, failures)
            if params is not None:
                self._latest_state = self._to_state_dict(params)
                torch.save(self._latest_state, SAVE_DIR / f"global_latest_{EXPERIMENT}.pth")
                print(f"[server/{EXPERIMENT}] round {server_round} aggregated "
                      f"{len(results)} updates", flush=True)
            return params, metrics

        def aggregate_evaluate(self, server_round, results, failures):
            loss, metrics = super().aggregate_evaluate(server_round, results, failures)
            auc = metrics.get("val_auc") if metrics else None
            if auc is not None:
                flag = ""
                if auc > self.best_auc:
                    self.best_auc = auc
                    if self._latest_state is not None:
                        torch.save(self._latest_state, SAVE_DIR / f"global_best_{EXPERIMENT}.pth")
                    flag = "  <-- new best"
                print(f"[server/{EXPERIMENT}] round {server_round} VAL mean AUC {auc:.4f} "
                      f"(best {self.best_auc:.4f}){flag}", flush=True)
                with self.log_path.open("a") as fh:
                    fh.write(f"{server_round},{auc:.5f},{loss if loss is not None else ''}\n")
            return loss, metrics

    return SaveModel


def build_strategy():
    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(
            [v.cpu().numpy() for v in get_model().state_dict().values()]
        ),
    )

    if STRATEGY == "fedavg":
        # No defense: every update, honest or not, is averaged in proportion to
        # the client's reported sample count.
        return make_saving_strategy(fl.server.strategy.FedAvg)(**common)

    # Multi-Krum: score each update by distance to its nearest neighbours and
    # average only the num_clients_to_keep most central ones, discarding outliers.
    return make_saving_strategy(fl.server.strategy.Krum)(
        num_malicious_clients=NUM_MALICIOUS,
        num_clients_to_keep=max(1, NUM_CLIENTS - NUM_MALICIOUS),
        **common,
    )


if __name__ == "__main__":
    print(f"--- Sentinel-AI Server | {STRATEGY.upper()} | {NUM_CLIENTS} clients "
          f"({NUM_MALICIOUS} malicious) | {NUM_ROUNDS} rounds ---", flush=True)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=build_strategy(),
    )

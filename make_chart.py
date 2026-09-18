"""Byzantine-robustness comparison chart for the Sentinel-AI report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = Path(__file__).resolve().parent / "data" / "models"

# Validated categorical slots 1-3 (light mode)
# (file, label, colour, label y-offset in points)
# The control and Multi-Krum curves land on the same value, so their end labels
# are staggered rather than overprinted.
SERIES = [
    ("history_clean_baseline.csv",   "No attack (control)",   "#2a78d6",  11),
    ("history_krum_attacked.csv",    "Attacked + Multi-Krum", "#eb6834", -11),
    ("history_fedavg_attacked.csv",  "Attacked + FedAvg",     "#1baf7a",   0),
]

SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def main():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for fname, label, color, dy in SERIES:
        path = MODELS / fname
        if not path.exists():
            print(f"skip (missing): {fname}")
            continue
        df = pd.read_csv(path)
        ax.plot(df["round"], df["val_auc"], color=color, linewidth=2,
                marker="o", markersize=4.5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=label, zorder=3, clip_on=False)
        # Direct label at the line end: identity is never colour-alone, and it
        # supplies the relief the contrast check requires.
        y = df["val_auc"].iloc[-1]
        ax.annotate(f"{label}  {y:.3f}", xy=(df['round'].iloc[-1], y),
                    xytext=(10, dy), textcoords="offset points", va="center",
                    fontsize=9, color=INK, zorder=4)

    ax.axhline(0.5, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 4)), zorder=1)
    ax.annotate("chance (AUC 0.5)", xy=(1, 0.5), xytext=(0, 6),
                textcoords="offset points", fontsize=8, color=INK_MUTED)

    ax.set_xlabel("Federated round", fontsize=10, color=INK_MUTED)
    ax.set_ylabel("Validation mean AUC (held-out patients)", fontsize=10, color=INK_MUTED)
    ax.set_title("One malicious hospital in five: Multi-Krum holds, FedAvg collapses",
                 fontsize=12.5, color=INK, pad=14, loc="left")

    ax.set_ylim(0.45, 0.87)
    ax.set_xlim(1, 15)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d7d2")
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    # Parked in the empty mid-left band so it never overlaps a series
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.52), frameon=False,
              fontsize=9, labelcolor=INK)

    out = MODELS.parent.parent / "byzantine_comparison.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

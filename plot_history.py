"""
Plot training curves (loss, accuracy, precision, recall) from metadata.json.

Usage:
    plot_history.bat
    plot_history.bat --metadata D:\\TF_CowDetection\\models\\metadata.json
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = os.environ.get("TF_COWDETECT_ROOT", HERE)
DEFAULT_META = os.path.join(EXTERNAL_DIR, "models", "metadata.json")
DEFAULT_OUTPUT = os.path.join(HERE, "training_curves.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--metadata", default=DEFAULT_META, help="Path to metadata.json")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("  TF_CowDetection  |  PLOT HISTORY")
    print("=" * 60)

    if not os.path.exists(args.metadata):
        print(f"ERROR: {args.metadata} not found. Run train.bat first.")
        return 1

    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    history = meta["history"]
    epochs = range(1, len(history["loss"]) + 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], "b-", label="Train")
    ax.plot(epochs, history["val_loss"], "r-", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Focal Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["accuracy"], "b-", label="Train")
    ax.plot(epochs, history["val_accuracy"], "r-", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision
    ax = axes[1, 0]
    ax.plot(epochs, history["precision"], "b-", label="Train")
    ax.plot(epochs, history["val_precision"], "r-", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title("Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recall
    ax = axes[1, 1]
    ax.plot(epochs, history["recall"], "b-", label="Train")
    ax.plot(epochs, history["val_recall"], "r-", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_title("Recall (cow detection rate)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Cow Detection U-Net  ({meta.get('train_size', '?')} train / "
                 f"{meta.get('val_size', '?')} val)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

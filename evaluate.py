"""
Evaluation for the transfer-learning stages.

    python evaluate.py --stage finetune
    python evaluate.py --stage pretrain --checkpoint path/to/best_model.pth

Produces, in outputs/<stage>/:
    classification_report.txt    overall + per-class metrics
    confusion_matrix.png         test-set confusion matrix
    per_class_accuracy.png       per-class accuracy bar chart
    training_curves.png          loss / accuracy per epoch
    eval_summary.json            machine-readable metrics

The reported split is named explicitly in every output: PlantDoc ships its own
train/test folders, so its metrics are a genuine held-out test result. A stage
evaluated on a randomly-split dataset is labelled "validation", not "test".
"""

import argparse
import json
import os

# ── Stage must be selected BEFORE config is imported ─────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    p.add_argument("--stage", default=None, choices=["pretrain", "finetune"])
    p.add_argument("--checkpoint", default=None, help="Explicit checkpoint path")
    p.add_argument("--dataset", default=None, help="Override dataset path")
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


ARGS = _parse_args()
if ARGS.stage:
    os.environ["PLANT_STAGE"] = ARGS.stage

import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import matplotlib                                                 # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from sklearn.metrics import (accuracy_score, precision_score,     # noqa: E402
                             recall_score, f1_score,
                             classification_report, confusion_matrix)

import config                                                     # noqa: E402

if ARGS.dataset:
    config.DATASET_ROOT = ARGS.dataset
if ARGS.img_size:
    config.IMG_SIZE = ARGS.img_size
if ARGS.batch_size:
    config.BATCH_SIZE = ARGS.batch_size

import data                                                       # noqa: E402
from model import build_model                                     # noqa: E402


# ─── Load ────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (ARGS.checkpoint or os.path.join(config.SAVE_DIR,
                                                       "best_model.pth"))
    if not os.path.exists(checkpoint_path):
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}\n"
                         f"Train it first:  python train.py --stage {config.ACTIVE_STAGE}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # The checkpoint carries everything needed to rebuild the model, so this
    # file can never drift out of sync with train.py.
    model = build_model(
        num_classes=ckpt["num_classes"],
        architecture=ckpt.get("architecture", "resnet50"),
        pretrained=False,
        init_weights=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    print(f"Loaded {ckpt.get('stage', '?')}/{ckpt.get('architecture', '?')} "
          f"from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']*100:.2f}%)")
    return model, ckpt, device


# ─── Charts ──────────────────────────────────────────────────────────────────
def plot_confusion(cm, names, split_name, acc, out_path):
    n = len(names)
    size = max(8, min(24, n * 0.85))
    fig, ax = plt.subplots(figsize=(size, size * 0.88))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"Confusion matrix — {split_name} (acc {acc:.4f})", fontsize=14)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)

    ticks = np.arange(n)
    labels = [name.split(" - ", 1)[-1][:26] for name in names]
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=max(5, 9 - n // 8))
    ax.set_yticklabels(labels, fontsize=max(5, 9 - n // 8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class(per_class, names, overall_acc, split_name, out_path):
    colors = ["#2ecc71" if a >= 0.90 else "#f39c12" if a >= 0.75 else "#e74c3c"
              for a in per_class]
    short = [n.split(" - ", 1)[-1] for n in names]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.55), 6.5))
    bars = ax.bar(short, per_class, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-class accuracy — {split_name}")
    ax.axhline(overall_acc, color="navy", linestyle="--", linewidth=1.5,
               label=f"Overall {overall_acc:.4f}")
    ax.legend()
    plt.xticks(rotation=60, ha="right", fontsize=7)

    for bar, acc in zip(bars, per_class):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curves(out_path):
    path = os.path.join(config.SAVE_DIR, "history.json")
    if not os.path.exists(path):
        print("  (no history.json — skipping training curves)")
        return
    with open(path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(epochs, h["train_loss"], label="Train", linewidth=2)
    ax1.plot(epochs, h["val_loss"], label="Validation", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss per epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in h["train_acc"]], label="Train", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in h["val_acc"]], label="Validation",
             linewidth=2, linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy per epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    model, ckpt, device = load_model()

    print("\n--- Loading data ---")
    _, test_loader, meta = data.get_dataloaders()
    idx_to_class = meta["idx_to_class"]
    num_classes = meta["num_classes"]
    class_names = [idx_to_class[i] for i in range(num_classes)]

    if num_classes != ckpt["num_classes"]:
        raise SystemExit(
            f"Class count mismatch: checkpoint has {ckpt['num_classes']}, "
            f"dataset has {num_classes}.\nYou are evaluating on the wrong "
            f"dataset for this checkpoint.")

    split_name = "test" if meta["uses_official_split"] else "validation"
    print(f"\n--- Running inference on the {split_name} split "
          f"({meta['num_val']} images) ---")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # labels=range(...) guarantees all classes appear in the report even if the
    # test split happens to contain none of them.
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(num_classes)),
        target_names=[n.split(" - ", 1)[-1] for n in class_names],
        digits=4, zero_division=0,
    )

    summary = (
        f"Split: {split_name}\n"
        f"Dataset: {meta.get('layout', '?')}"
        f"{' (official train/test)' if meta['uses_official_split'] else ' (random split)'}\n"
        f"Samples: {len(all_labels)}\n\n"
        f"Test Results:\n"
        f"Accuracy: {acc:.4f}\n"
        f"Precision (weighted): {prec:.4f}\n"
        f"Recall (weighted): {rec:.4f}\n"
        f"F1-score (weighted): {f1:.4f}\n\n"
        f"Classification Report:\n{report}"
    )
    print("\n" + summary)

    txt_path = os.path.join(config.SAVE_DIR, "classification_report.txt")
    with open(txt_path, "w") as f:
        f.write(summary)

    # ── Charts ──
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    cm_path = os.path.join(config.SAVE_DIR, "confusion_matrix.png")
    plot_confusion(cm, class_names, split_name, acc, cm_path)

    per_class = []
    for i in range(num_classes):
        mask = all_labels == i
        per_class.append(float((all_preds[mask] == i).mean()) if mask.sum() else 0.0)
    pc_path = os.path.join(config.SAVE_DIR, "per_class_accuracy.png")
    plot_per_class(per_class, class_names, acc, split_name, pc_path)

    curves_path = os.path.join(config.SAVE_DIR, "training_curves.png")
    plot_curves(curves_path)

    # ── Machine-readable ──
    eval_summary = {
        "stage": ckpt.get("stage", config.ACTIVE_STAGE),
        "architecture": ckpt.get("architecture", "?"),
        "checkpoint_epoch": ckpt["epoch"],
        "metric_split": split_name,
        "dataset": config.DATASET_ROOT,
        "layout": meta.get("layout", "?"),
        "uses_official_split": meta["uses_official_split"],
        "num_samples": int(len(all_labels)),
        "num_classes": num_classes,
        "accuracy": float(acc),
        "precision_weighted": float(prec),
        "recall_weighted": float(rec),
        "f1_weighted": float(f1),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro",
                                   zero_division=0)),
        "per_class_accuracy": dict(zip(class_names, per_class)),
        "weakest_classes": sorted(
            [(n.split(" - ", 1)[-1], round(a, 4))
             for n, a in zip(class_names, per_class)], key=lambda x: x[1])[:5],
    }
    json_path = os.path.join(config.SAVE_DIR, "eval_summary.json")
    with open(json_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("\nSaved:")
    for path in (txt_path, cm_path, pc_path, curves_path, json_path):
        if os.path.exists(path):
            print(f"  {path}")

    weakest = eval_summary["weakest_classes"]
    print(f"\nWeakest classes on the {split_name} split:")
    for name, value in weakest:
        print(f"  {name:<40s} {value*100:5.1f}%")


if __name__ == "__main__":
    main()

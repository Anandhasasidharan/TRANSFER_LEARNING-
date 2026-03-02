"""
Standalone Evaluation Script
Loads best_model.pth and generates:
  - Classification Report (txt)
  - Confusion Matrix (png)
  - Per-class Accuracy Bar Chart (png)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
from PIL import Image

# ─────────────────────────────────────────────
#  CONFIG  (must match train.py)
# ─────────────────────────────────────────────
DATA_DIR   = r"D:\college\DATASET\Augmented_Dataset"
SAVE_DIR   = r"D:\college\MININ_PROJECT_CODE\TRANSFER_LEARNING\outputs"
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
BATCH_SIZE = 64
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.10
SEED       = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ─────────────────────────────────────────────
#  DATASET (same split logic as train.py)
# ─────────────────────────────────────────────
class MultiCropDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {}
        self.classes = []

        crop_dirs = sorted([d for d in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, d))])
        for crop in crop_dirs:
            crop_path = os.path.join(root_dir, crop)
            disease_dirs = sorted([d for d in os.listdir(crop_path)
                                   if os.path.isdir(os.path.join(crop_path, d))])
            for disease in disease_dirs:
                label_name = f"{crop}__{disease}"
                if label_name not in self.class_to_idx:
                    self.class_to_idx[label_name] = len(self.classes)
                    self.classes.append(label_name)
                label_idx = self.class_to_idx[label_name]
                disease_path = os.path.join(crop_path, disease)
                for fname in os.listdir(disease_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        self.samples.append((os.path.join(disease_path, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

print("\n--- Loading Dataset ---")
full_dataset = MultiCropDataset(DATA_DIR)
num_total    = len(full_dataset)
num_val      = int(num_total * VAL_SPLIT)
num_test     = int(num_total * TEST_SPLIT)
num_train    = num_total - num_val - num_test

_, _, test_ds = random_split(
    full_dataset,
    [num_train, num_val, num_test],
    generator=torch.Generator().manual_seed(SEED)
)

test_ds_t   = TransformSubset(test_ds, val_test_transforms)
test_loader = DataLoader(test_ds_t, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

num_classes = len(full_dataset.classes)
short_names = [c.split("__", 1)[1] for c in full_dataset.classes]
print(f"Test samples : {len(test_ds_t)}")
print(f"Classes      : {num_classes}")

# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
print(f"\n--- Loading Model from {MODEL_PATH} ---")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print("Model loaded successfully.")

# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────
print("\n--- Running Inference on Test Set ---")
all_preds  = []
all_labels = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        if i % 10 == 0:
            print(f"  batch {i}/{len(test_loader)}")

# ─────────────────────────────────────────────
#  METRICS & CLASSIFICATION REPORT
# ─────────────────────────────────────────────
test_acc       = accuracy_score(all_labels, all_preds)
test_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
test_recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
test_f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
report         = classification_report(all_labels, all_preds, target_names=short_names, zero_division=0)

summary = (
    f"Test Results:\n"
    f"Accuracy: {test_acc:.4f}\n"
    f"Precision (weighted): {test_precision:.4f}\n"
    f"Recall (weighted): {test_recall:.4f}\n"
    f"F1-score (weighted): {test_f1:.4f}\n\n"
    f"Classification Report:\n{report}"
)
print("\n" + summary)

report_path = os.path.join(SAVE_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(summary)
print(f"Saved: {report_path}")

# ─────────────────────────────────────────────
#  CONFUSION MATRIX
# ─────────────────────────────────────────────
print("\n--- Generating Confusion Matrix ---")
cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(22, 18))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=short_names, yticklabels=short_names,
            linewidths=0.5, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=13)
ax.set_ylabel("True Label", fontsize=13)
ax.set_title(f"Confusion Matrix  (Test Acc = {test_acc:.4f})", fontsize=15)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"Saved: {cm_path}")

# ─────────────────────────────────────────────
#  PER-CLASS ACCURACY BAR CHART
# ─────────────────────────────────────────────
print("\n--- Generating Per-Class Accuracy Chart ---")
all_labels_np = np.array(all_labels)
all_preds_np  = np.array(all_preds)
per_class_acc = []
for i in range(num_classes):
    mask = all_labels_np == i
    if mask.sum() > 0:
        per_class_acc.append((all_preds_np[mask] == i).mean())
    else:
        per_class_acc.append(0.0)

colors = ["#2ecc71" if a >= 0.90 else "#f39c12" if a >= 0.75 else "#e74c3c"
          for a in per_class_acc]

fig, ax = plt.subplots(figsize=(18, 8))
bars = ax.bar(short_names, per_class_acc, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Per-Class Test Accuracy", fontsize=14)
ax.axhline(y=test_acc, color="navy", linestyle="--", linewidth=1.5, label=f"Overall Acc: {test_acc:.4f}")
ax.legend(fontsize=11)
plt.xticks(rotation=45, ha="right", fontsize=8)

for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.2f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
bar_path = os.path.join(SAVE_DIR, "per_class_accuracy.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"Saved: {bar_path}")

# ─────────────────────────────────────────────
#  TRAINING CURVES (if history.json exists)
# ─────────────────────────────────────────────
history_path = os.path.join(SAVE_DIR, "history.json")
if os.path.exists(history_path):
    print("\n--- Generating Training Curves from history.json ---")
    with open(history_path) as f:
        history = json.load(f)
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend()
    axes[1].plot(epochs_range, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs_range, history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].legend()
    plt.tight_layout()
    curves_path = os.path.join(SAVE_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Saved: {curves_path}")

print("\n✅ All outputs saved to:", SAVE_DIR)

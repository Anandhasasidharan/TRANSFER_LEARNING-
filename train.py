"""
Multi-Crop Disease Classification using Transfer Learning (ResNet-50)
Crops: Black Gram, Cotton, Soybean
Total Classes: 33
"""

import os
import time
import copy
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = r"D:\college\DATASET\Augmented_Dataset"
SAVE_DIR    = r"D:\college\MININ_PROJECT_CODE\TRANSFER_LEARNING\outputs"
NUM_EPOCHS  = 40
BATCH_SIZE  = 64
LR          = 0.001
STEP_SIZE   = 10         # decay LR every N epochs
GAMMA       = 0.1        # LR decay factor
VAL_SPLIT   = 0.15       # 15 % validation
TEST_SPLIT  = 0.10       # 10 % test
SEED        = 42
NUM_WORKERS = 0

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(SEED)

# ─────────────────────────────────────────────
#  DATA TRANSFORMS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
#  FLAT MULTI-CROP DATASET
#  We merge all crop sub-folders into one flat
#  dataset where the class name = "Crop_Disease"
# ─────────────────────────────────────────────
class MultiCropDataset(torch.utils.data.Dataset):
    """Merge multiple ImageFolder roots into one dataset with prefixed labels."""

    def __init__(self, root_dir, transform=None):
        self.samples = []          # list of (path, label_idx)
        self.transform = transform
        self.class_to_idx = {}
        self.classes = []

        crop_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for crop in crop_dirs:
            crop_path = os.path.join(root_dir, crop)
            disease_dirs = sorted([
                d for d in os.listdir(crop_path)
                if os.path.isdir(os.path.join(crop_path, d))
            ])
            for disease in disease_dirs:
                label_name = f"{crop}__{disease}"   # e.g. Black_Gram_cleaned__anthracnose
                if label_name not in self.class_to_idx:
                    self.class_to_idx[label_name] = len(self.classes)
                    self.classes.append(label_name)

                label_idx = self.class_to_idx[label_name]
                disease_path = os.path.join(crop_path, disease)

                for fname in os.listdir(disease_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        self.samples.append((os.path.join(disease_path, fname), label_idx))

        print(f"Total images : {len(self.samples)}")
        print(f"Total classes: {len(self.classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
#  LOAD & SPLIT DATA
# ─────────────────────────────────────────────
print("\n--- Loading Dataset ---")
full_dataset = MultiCropDataset(DATA_DIR, transform=None)

num_total = len(full_dataset)
num_val   = int(num_total * VAL_SPLIT)
num_test  = int(num_total * TEST_SPLIT)
num_train = num_total - num_val - num_test

train_ds, val_ds, test_ds = random_split(
    full_dataset,
    [num_train, num_val, num_test],
    generator=torch.Generator().manual_seed(SEED)
)

# Apply transforms via wrapper
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        from PIL import Image
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


train_ds_t = TransformSubset(train_ds, train_transforms)
val_ds_t   = TransformSubset(val_ds,   val_test_transforms)
test_ds_t  = TransformSubset(test_ds,  val_test_transforms)

train_loader = DataLoader(train_ds_t, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds_t,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds_t,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

num_classes = len(full_dataset.classes)
print(f"\nTrain: {len(train_ds_t)} | Val: {len(val_ds_t)} | Test: {len(test_ds_t)}")
print(f"Number of classes: {num_classes}")

# Save class names
with open(os.path.join(SAVE_DIR, "class_names.json"), "w") as f:
    json.dump(full_dataset.classes, f, indent=2)
print(f"Class names saved to {SAVE_DIR}\\class_names.json")

# ─────────────────────────────────────────────
#  MODEL — ResNet-50 with Fine-Tuning
# ─────────────────────────────────────────────
print("\n--- Building Model ---")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer3, layer4, and fc for fine-tuning
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final FC layer
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# Count trainable parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}")

# ─────────────────────────────────────────────
#  LOSS, OPTIMIZER, SCHEDULER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}  LR={scheduler.get_last_lr()}")
        print("-" * 50)
        t0 = time.time()

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if batch_idx % 50 == 0:
                    print(f"  [{phase}] batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}")

            epoch_loss = running_loss     / (len(train_ds_t) if phase == "train" else len(val_ds_t))
            epoch_acc  = running_corrects.double() / (len(train_ds_t) if phase == "train" else len(val_ds_t))

            print(f"  {phase.upper()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
                print(f"  *** Best model saved (acc={best_acc:.4f}) ***")

        scheduler.step()
        print(f"  Epoch time: {time.time()-t0:.1f}s")

    print(f"\nBest Val Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


print("\n--- Starting Training ---")
model, history = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)

# ─────────────────────────────────────────────
#  TEST EVALUATION
# ─────────────────────────────────────────────
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, classification_report)

print("\n--- Test Evaluation ---")
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Short class names (strip crop prefix for readability)
short_names = [c.split("__", 1)[1] for c in full_dataset.classes]

test_acc       = accuracy_score(all_labels, all_preds)
test_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
test_recall    = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
test_f1        = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

report = classification_report(all_labels, all_preds, target_names=short_names, zero_division=0)

summary = (
    f"Test Results:\n"
    f"Accuracy: {test_acc:.4f}\n"
    f"Precision (weighted): {test_precision:.4f}\n"
    f"Recall (weighted): {test_recall:.4f}\n"
    f"F1-score (weighted): {test_f1:.4f}\n\n"
    f"Classification Report:\n{report}"
)

print(summary)

# Save to file
report_path = os.path.join(SAVE_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(summary)
print(f"Classification report saved to {report_path}")

# ─────────────────────────────────────────────
#  SAVE TRAINING CURVES
# ─────────────────────────────────────────────
epochs_range = range(1, NUM_EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss")
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epochs_range, history["train_acc"], label="Train Acc")
axes[1].plot(epochs_range, history["val_acc"],   label="Val Acc")
axes[1].set_title("Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
print(f"Training curves saved to {SAVE_DIR}\\training_curves.png")

# Save history
with open(os.path.join(SAVE_DIR, "history.json"), "w") as f:
    json.dump(history, f, indent=2)

print("\nDone! Outputs saved to:", SAVE_DIR)

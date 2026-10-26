"""
Dataset indexing, splitting and DataLoaders for the 3-stage pipeline.

Auto-detected on-disk layouts
-----------------------------
nested     Crop/Disease/*.jpg          label -> "Crop - Disease"
flat       ClassName/*.jpg             label -> derived from the folder name
presplit   train/ClassName/*.jpg       the dataset's own train/test split is
           + test/ClassName/*.jpg      reused as-is (no random split, so no
                                       near-duplicate leakage into validation)

The "Crop - Disease" label format is what evaluate.py expects when it builds
its per-crop reports, so nested and flat sources both produce it.

Why the split matters: a random split of a field dataset leaks near-duplicate
frames of the same leaf into validation and inflates accuracy by 10-15 points.
When a dataset ships an official split we always use it.
"""

import os
import re
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import config


# ─── Crop aliases (legacy folder names -> display names) ─────────────────────
CROP_ALIASES = {
    "Black_Gram_cleaned": "Black_Gram",
    "Cleaned_Cotten_Final": "Cotton",
    "soyabean_final": "Soybean",
}

# Crop names used to recover the crop from flat class folders such as
# "Apple Scab Leaf" or "Tomato leaf late blight" (PlantDoc style).
KNOWN_CROPS = [
    "Apple", "Bell_pepper", "Blueberry", "Cherry", "Corn", "Grape", "Lemon",
    "Mango", "Peach", "Pepper_bell", "Potato", "Raspberry", "Soyabean",
    "Soybean", "Squash", "Strawberry", "Tomato", "Rice", "Cotton",
    "Black_Gram", "Wheat", "Sugarcane", "Cassava",
]


# ─── Layout detection ────────────────────────────────────────────────────────
def _image_files(folder: str):
    """Image files directly inside `folder` (sorted, deterministic)."""
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if os.path.splitext(f)[1].lower() in config.VALID_EXTENSIONS
    ]


def _subdirs(folder: str):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, d)
        for d in sorted(os.listdir(folder))
        if os.path.isdir(os.path.join(folder, d))
    ]


def _images_recursive(folder: str, max_depth: int = 2):
    """Images in `folder`, descending up to `max_depth` extra levels."""
    found = _image_files(folder)
    if found:
        return found
    if max_depth <= 0:
        return []
    for sub in _subdirs(folder):
        found.extend(_images_recursive(sub, max_depth - 1))
    return found


def detect_layout(dataset_root: str) -> str:
    """Return one of: "presplit", "nested", "flat"."""
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            f"Set CNN_DATASET_DIR or run prepare_datasets.py first."
        )

    top = _subdirs(dataset_root)
    if not top:
        raise ValueError(f"No sub-folders in {dataset_root}")

    names = {os.path.basename(p).lower(): p for p in top}

    train_dir = names.get("train")
    val_dir = names.get("test") or names.get("val") or names.get("valid")
    if train_dir and val_dir:
        return "presplit"

    # A single wrapper folder (e.g. PlantDoc-Dataset/PlantDoc-Dataset/...) —
    # descend once rather than mis-detecting.
    if len(top) == 1:
        inner = _subdirs(top[0])
        inner_names = {os.path.basename(p).lower() for p in inner}
        if {"train", "test"} & inner_names:
            return "presplit"

    for path in top:
        if _image_files(path):
            return "flat"
        if _images_recursive(path, max_depth=2):
            return "nested"

    raise ValueError(f"Could not find any images under {dataset_root}")


# ─── Label construction ──────────────────────────────────────────────────────
def _guess_crop(name: str):
    """Recover a crop name from a flat class folder name."""
    low = name.lower()
    best = None
    for crop in KNOWN_CROPS:
        if low.startswith(crop.lower()) and (best is None or len(crop) > len(best)):
            best = crop
    if best:
        return best
    # Fall back to whatever precedes the word "leaf"
    m = re.match(r"^(.*?)\s+leaf\b", name, flags=re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip().replace(" ", "_")
    return None


def _flat_label(folder_name: str):
    """Build (crop, label) for a flat class folder."""
    if "___" in folder_name:
        crop, disease = folder_name.split("___", 1)
        crop, disease = crop.strip(), disease.strip()
        return crop, f"{crop} - {disease}"
    if " - " in folder_name:
        return folder_name.split(" - ", 1)[0], folder_name
    crop = _guess_crop(folder_name)
    if crop:
        return crop, f"{crop} - {folder_name}"
    return None, folder_name


# ─── Scanning ────────────────────────────────────────────────────────────────
def _new_index(layout: str):
    return {
        "layout": layout,
        "class_to_idx": {},
        "idx_to_class": {},
        "crop_to_idx": {},
        "train_samples": [],
        "val_samples": [],
        "uses_official_split": layout == "presplit",
    }


def _scan_nested(root: str, index: dict):
    for crop_path in _subdirs(root):
        crop_name = CROP_ALIASES.get(os.path.basename(crop_path), os.path.basename(crop_path))
        for disease_path in _subdirs(crop_path):
            images = _images_recursive(disease_path)
            if not images:
                continue
            label = f"{crop_name} - {os.path.basename(disease_path)}"
            if label not in index["class_to_idx"]:
                idx = len(index["class_to_idx"])
                index["class_to_idx"][label] = idx
                index["idx_to_class"][idx] = label
            idx = index["class_to_idx"][label]
            if crop_name not in index["crop_to_idx"]:
                index["crop_to_idx"][crop_name] = len(index["crop_to_idx"])
            for img in images:
                index["train_samples"].append((img, idx))


def _scan_flat(root: str, index: dict, into: str = "train_samples"):
    for class_path in _subdirs(root):
        images = _images_recursive(class_path)
        if not images:
            continue
        crop, label = _flat_label(os.path.basename(class_path))
        if label not in index["class_to_idx"]:
            idx = len(index["class_to_idx"])
            index["class_to_idx"][label] = idx
            index["idx_to_class"][idx] = label
        idx = index["class_to_idx"][label]
        if crop and crop not in index["crop_to_idx"]:
            index["crop_to_idx"][crop] = len(index["crop_to_idx"])
        for img in images:
            index[into].append((img, idx))


def _find_split_dirs(root: str):
    """Return (train_dir, val_dir) for a presplit dataset, descending if wrapped."""
    def _pick(base):
        subs = {os.path.basename(p).lower(): p for p in _subdirs(base)}
        train = subs.get("train")
        val = subs.get("test") or subs.get("val") or subs.get("valid")
        return train, val

    train, val = _pick(root)
    if train and val:
        return train, val
    for sub in _subdirs(root):
        train, val = _pick(sub)
        if train and val:
            return train, val
    raise ValueError(f"No train/test split found under {root}")


def _scan_presplit(root: str, index: dict):
    train_dir, val_dir = _find_split_dirs(root)
    # Build one shared class index from the training folder, then add any
    # classes that only appear in the val folder so indices stay consistent.
    _scan_flat(train_dir, index, into="train_samples")
    tmp = {"class_to_idx": {}, "idx_to_class": {}, "crop_to_idx": {}, "val_samples": []}
    _scan_flat(val_dir, tmp, into="val_samples")
    remap = {}
    for label, old_idx in tmp["class_to_idx"].items():
        if label not in index["class_to_idx"]:
            new_idx = len(index["class_to_idx"])
            index["class_to_idx"][label] = new_idx
            index["idx_to_class"][new_idx] = label
        remap[old_idx] = index["class_to_idx"][label]
    for crop in tmp["crop_to_idx"]:
        index["crop_to_idx"].setdefault(crop, len(index["crop_to_idx"]))
    index["val_samples"] = [(p, remap[l]) for p, l in tmp["val_samples"]]

    train_classes = {l for _, l in index["train_samples"]}
    val_classes = {l for _, l in index["val_samples"]}
    orphans = val_classes - train_classes
    if orphans:
        names = [index["idx_to_class"][i] for i in sorted(orphans)]
        print(f"  WARNING: {len(orphans)} class(es) appear only in the val split:")
        for n in names[:10]:
            print(f"    - {n}")
        print("  These can never be predicted correctly — check the split.")


# ─── Split & index ───────────────────────────────────────────────────────────
def split_dataset(samples, train_ratio=config.TRAIN_RATIO, seed=config.RANDOM_SEED):
    """Stratified train/val split.

    Classes with fewer than 2 samples are placed entirely into the training
    set (stratified splitting needs at least 2 members per class).
    """
    from sklearn.model_selection import train_test_split

    label_counts = Counter(label for _, label in samples)
    safe = [(p, l) for p, l in samples if label_counts[l] >= 2]
    rare = [(p, l) for p, l in samples if label_counts[l] < 2]

    if rare:
        rare_classes = {l for _, l in rare}
        print(f"  Note: {len(rare)} image(s) from {len(rare_classes)} rare class(es) "
              f"added directly to the train set.")

    paths, labels = zip(*safe)
    paths, labels = list(paths), list(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        paths, labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=seed,
    )
    return list(zip(X_train, y_train)) + rare, list(zip(X_val, y_val))


def _warn_if_preaugmented(dataset_root: str, layout: str):
    """Offline-augmented copies of the same leaf must not be randomly split."""
    if layout == "presplit":
        return
    if "augment" in dataset_root.lower():
        print("\n" + "!" * 78)
        print("  WARNING: dataset path looks like an offline-augmented copy")
        print(f"           ({dataset_root})")
        print("  A random split will put augmented copies of the SAME leaf in")
        print("  both train and val, inflating accuracy by 10-15 points.")
        print("  Split by original image first, or use a dataset with an")
        print("  official train/test folder. Continuing anyway...")
        print("!" * 78 + "\n")


def scan_dataset(dataset_root: str = None, layout: str = "auto"):
    """Index a dataset. Returns a dict describing classes and samples."""
    dataset_root = dataset_root or config.DATASET_ROOT
    layout = detect_layout(dataset_root) if layout == "auto" else layout
    _warn_if_preaugmented(dataset_root, layout)

    index = _new_index(layout)
    index["root"] = dataset_root

    if layout == "presplit":
        _scan_presplit(dataset_root, index)
    elif layout == "flat":
        _scan_flat(dataset_root, index)
    else:
        _scan_nested(dataset_root, index)

    if not index["uses_official_split"]:
        index["train_samples"], index["val_samples"] = split_dataset(index["train_samples"])

    if not index["train_samples"]:
        raise ValueError(f"No training images found in {dataset_root} (layout={layout})")

    return index


# Backwards-compatible alias for code written against the old API.
def build_class_mapping(dataset_root: str):
    idx = scan_dataset(dataset_root)
    crops = [idx["crop_to_idx"].get(c, 0) for c, _ in idx["train_samples"]]
    return (idx["class_to_idx"], idx["idx_to_class"],
            idx["train_samples"], idx["crop_to_idx"], crops)


# ─── Transforms ──────────────────────────────────────────────────────────────
def get_train_transforms(img_size: int = None):
    img_size = img_size or config.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])


def get_val_transforms(img_size: int = None):
    img_size = img_size or config.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ─── Dataset ─────────────────────────────────────────────────────────────────
class CropDiseaseDataset(Dataset):
    """PyTorch Dataset for crop disease images."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ─── Class balancing ─────────────────────────────────────────────────────────
def make_weighted_sampler(samples):
    """WeightedRandomSampler to counter class imbalance."""
    labels = [s[1] for s in samples]
    class_counts = Counter(labels)
    total = len(labels)
    class_weights = {cls: total / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─── Statistics (used by report.py / prepare_datasets.py) ────────────────────
def dataset_stats(dataset_root: str = None, layout: str = "auto"):
    """Count images per class without building DataLoaders."""
    index = scan_dataset(dataset_root, layout=layout)
    train_counts = Counter(l for _, l in index["train_samples"])
    val_counts = Counter(l for _, l in index["val_samples"])

    per_class = []
    for idx in sorted(index["idx_to_class"]):
        name = index["idx_to_class"][idx]
        tr, va = train_counts.get(idx, 0), val_counts.get(idx, 0)
        per_class.append({
            "label": name,
            "crop": name.split(" - ")[0],
            "train": tr,
            "val": va,
            "total": tr + va,
        })

    totals = [c["total"] for c in per_class]
    return {
        "root": index["root"],
        "layout": index["layout"],
        "uses_official_split": index["uses_official_split"],
        "num_classes": len(per_class),
        "num_crops": len(index["crop_to_idx"]),
        "crops": sorted(index["crop_to_idx"].keys()),
        "num_train": len(index["train_samples"]),
        "num_val": len(index["val_samples"]),
        "num_total": len(index["train_samples"]) + len(index["val_samples"]),
        "per_class": per_class,
        "imbalance_ratio": (max(totals) / min(totals)) if totals and min(totals) else float("inf"),
        "min_class": min(totals) if totals else 0,
        "max_class": max(totals) if totals else 0,
    }


# ─── DataLoaders ─────────────────────────────────────────────────────────────
def get_dataloaders(dataset_root: str = None, batch_size: int = None,
                    img_size: int = None):
    """Build train / val DataLoaders plus metadata."""
    dataset_root = dataset_root or config.DATASET_ROOT
    batch_size = batch_size or config.BATCH_SIZE
    img_size = img_size or config.IMG_SIZE

    index = scan_dataset(dataset_root)
    num_classes = len(index["class_to_idx"])

    print("Dataset Summary:")
    print(f"  Root         : {index['root']}")
    print(f"  Layout       : {index['layout']}"
          + ("  (official train/test split)" if index["uses_official_split"] else "  (random stratified split)"))
    print(f"  Total images : {len(index['train_samples']) + len(index['val_samples'])}")
    print(f"  Num classes  : {num_classes}")
    print(f"  Crops        : {', '.join(sorted(index['crop_to_idx'])) or '-'}")
    print(f"  Train / Val  : {len(index['train_samples'])} / {len(index['val_samples'])}")
    print()

    train_ds = CropDiseaseDataset(index["train_samples"], transform=get_train_transforms(img_size))
    val_ds = CropDiseaseDataset(index["val_samples"], transform=get_val_transforms(img_size))

    sampler = make_weighted_sampler(index["train_samples"]) if config.USE_WEIGHTED_SAMPLER else None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        drop_last=len(train_ds) > batch_size,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
    )

    meta = {
        "num_classes": num_classes,
        "class_to_idx": index["class_to_idx"],
        "idx_to_class": index["idx_to_class"],
        "crop_to_idx": index["crop_to_idx"],
        "train_samples": index["train_samples"],
        "val_samples": index["val_samples"],
        "layout": index["layout"],
        "uses_official_split": index["uses_official_split"],
        "num_train": len(index["train_samples"]),
        "num_val": len(index["val_samples"]),
    }
    return train_loader, val_loader, meta


if __name__ == "__main__":
    import json
    stats = dataset_stats()
    print(json.dumps({k: v for k, v in stats.items() if k != "per_class"}, indent=2))
    for row in stats["per_class"]:
        print(f"  {row['label']:<55s} {row['train']:>6d} / {row['val']:>5d}")

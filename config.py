"""
Configuration for transfer learning on PlantVillage -> PlantDoc.

Two stages, both ResNet-50:

  pretrain   ImageNet weights fine-tuned on the full PlantVillage dataset
             (54k lab images, 38 classes). Builds leaf-domain features.

  finetune   Those weights transferred onto PlantDoc (2.6k real field photos,
             27 classes). This is the headline result.

Usage
-----
    python train.py --stage pretrain
    python train.py --stage finetune

Dataset location
----------------
Both datasets live under one root, overridable with an environment variable:

    set PLANT_DATA_DIR=D:\\AI\\DATASET

Expected layout:

    <PLANT_DATA_DIR>/PlantVillage-Dataset/color/Potato___Early_blight/*.JPG
    <PLANT_DATA_DIR>/PlantDoc-Dataset/train/<class>/*.jpg
    <PLANT_DATA_DIR>/PlantDoc-Dataset/test/<class>/*.jpg

Get them with:
    git clone https://github.com/spMohanty/PlantVillage-Dataset
    git clone https://github.com/pratikkayal/PlantDoc-Dataset
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Root holding PlantVillage-Dataset/ and PlantDoc-Dataset/
DATA_DIR = os.environ.get("PLANT_DATA_DIR", r"D:\AI\DATASET")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

SEED = 42
RANDOM_SEED = SEED     # alias used by data.py
NUM_WORKERS = 0        # 0 is safest on Windows
PIN_MEMORY = True

TRAIN_RATIO = 0.80     # only used for datasets without an official split
VAL_RATIO = 0.20


STAGES = {
    "pretrain": {
        "dataset_subdir": os.path.join("PlantVillage-Dataset", "color"),
        "architecture": "resnet50",
        "pretrained": True,
        "init_from": None,
        "img_size": 224,
        "batch_size": 64,
        "num_epochs": 12,
        "head_lr": 3e-4,
        "backbone_lr": 3e-5,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "label_smoothing": 0.1,
        "freeze_backbone_epochs": 1,
        "scheduler": "cosine",
        "step_lr_size": 10,
        "step_lr_gamma": 0.5,
        "warmup_epochs": 1,
        "early_stop_patience": 5,
        "use_weighted_sampler": True,
        "num_workers": 0,
        "num_threads": 8,       # torch.set_num_threads, helps on Windows
        "amp": True,
    },
    "finetune": {
        "dataset_subdir": "PlantDoc-Dataset",
        "architecture": "resnet50",
        "pretrained": False,
        "init_from": "pretrain",
        "img_size": 224,
        "batch_size": 32,        # smaller batch keeps BatchNorm stats sane
        "num_epochs": 25,
        "head_lr": 3e-4,
        "backbone_lr": 1e-5,     # ~30x lower than the head
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "label_smoothing": 0.1,
        "freeze_backbone_epochs": 3,
        "scheduler": "cosine",
        "step_lr_size": 10,
        "step_lr_gamma": 0.5,
        "warmup_epochs": 1,
        "early_stop_patience": 6,
        "use_weighted_sampler": True,
        "num_workers": 0,
        "num_threads": 8,
        "amp": True,
    },
}

DEFAULT_STAGE = "pretrain"


def get_stage() -> str:
    stage = os.environ.get("PLANT_STAGE", DEFAULT_STAGE).strip().lower()
    if stage not in STAGES:
        raise SystemExit(f"Unknown stage {stage!r}. Choose: {', '.join(STAGES)}")
    return stage


def _resolve_init(value):
    if not value:
        return None
    if value in STAGES:
        return os.path.join(BASE_DIR, "outputs", value, "best_model.pth")
    return value


ACTIVE_STAGE = get_stage()
_ACTIVE = STAGES[ACTIVE_STAGE]

DATASET_ROOT = os.path.join(DATA_DIR, _ACTIVE["dataset_subdir"])
IMG_SIZE = _ACTIVE["img_size"]

ARCHITECTURE = _ACTIVE["architecture"]
PRETRAINED = _ACTIVE["pretrained"]
INIT_CHECKPOINT = _resolve_init(_ACTIVE["init_from"])
DROPOUT = _ACTIVE["dropout"]

BATCH_SIZE = _ACTIVE["batch_size"]
NUM_EPOCHS = _ACTIVE["num_epochs"]
HEAD_LR = _ACTIVE["head_lr"]
BACKBONE_LR = _ACTIVE["backbone_lr"]
LR = HEAD_LR                       # alias
WEIGHT_DECAY = _ACTIVE["weight_decay"]
LABEL_SMOOTHING = _ACTIVE["label_smoothing"]
FREEZE_BACKBONE_EPOCHS = _ACTIVE["freeze_backbone_epochs"]
SCHEDULER = _ACTIVE["scheduler"]
STEP_LR_SIZE = _ACTIVE["step_lr_size"]
STEP_LR_GAMMA = _ACTIVE["step_lr_gamma"]
WARMUP_EPOCHS = _ACTIVE["warmup_epochs"]
EARLY_STOP_PATIENCE = _ACTIVE["early_stop_patience"]
USE_WEIGHTED_SAMPLER = _ACTIVE["use_weighted_sampler"]
AMP = _ACTIVE["amp"]

SAVE_DIR = os.path.join(BASE_DIR, "outputs", ACTIVE_STAGE)
os.makedirs(SAVE_DIR, exist_ok=True)

"""
Transfer learning on PlantVillage -> PlantDoc with ResNet-50.

    python train.py --stage pretrain      # ImageNet -> PlantVillage (54k, 38 cls)
    python train.py --stage finetune      # PlantVillage -> PlantDoc (2.6k, 27 cls)

Two-phase fine-tuning: the classifier head trains alone for the first N epochs,
then the backbone unfreezes at a much lower learning rate. Unfreezing everything
at lr=0.001 (the previous setting) overwrites the pretrained filters within one
epoch, which is why a transfer-learning model can end up worse than scratch.

Data layout and stage settings live in config.py and data.py.
"""

import argparse
import csv
import json
import math
import os
import time

# ── Stage must be selected BEFORE config is imported ─────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Train a transfer-learning stage.")
    p.add_argument("--stage", default=None, choices=["pretrain", "finetune"])
    p.add_argument("--arch", default=None, help="e.g. resnet50, resnet18, efficientnet_b0")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--data-dir", default=None, help="Root holding the datasets")
    p.add_argument("--dataset", default=None, help="Override the stage dataset path")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-sampler", action="store_true")
    p.add_argument("--no-init", action="store_true",
                   help="Ignore init_from and start from ImageNet weights only")
    return p.parse_args()


ARGS = _parse_args()
if ARGS.stage:
    os.environ["PLANT_STAGE"] = ARGS.stage
if ARGS.data_dir:
    os.environ["PLANT_DATA_DIR"] = ARGS.data_dir

import config                                                    # noqa: E402

if ARGS.epochs:
    config.NUM_EPOCHS = ARGS.epochs
if ARGS.batch_size:
    config.BATCH_SIZE = ARGS.batch_size
if ARGS.img_size:
    config.IMG_SIZE = ARGS.img_size
if ARGS.dataset:
    config.DATASET_ROOT = ARGS.dataset
if ARGS.arch:
    config.ARCHITECTURE = ARGS.arch
if ARGS.num_workers is not None:
    config.NUM_WORKERS = ARGS.num_workers
if ARGS.no_sampler:
    config.USE_WEIGHTED_SAMPLER = False
if ARGS.no_init:
    config.INIT_CHECKPOINT = None
    config.PRETRAINED = True

import torch                                                     # noqa: E402
import torch.nn as nn                                            # noqa: E402
from torch.amp import GradScaler, autocast                        # noqa: E402
from tqdm import tqdm                                             # noqa: E402

import data                                                       # noqa: E402
from model import (build_model, set_backbone_trainable,           # noqa: E402
                   set_backbone_bn_eval, head_parameters,
                   backbone_parameters)


def accuracy(output, target):
    _, preds = output.max(1)
    return preds.eq(target).sum().item() / target.size(0)


def banner(text, width=78):
    print(f"\n{'=' * width}\n  {text}\n{'=' * width}")


def make_scheduler(optimizer, total_epochs, warmup_epochs, kind,
                   step_size, gamma):
    """Linear warm-up followed by cosine or step decay, in one scheduler."""
    def schedule(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        if kind == "cosine":
            span = max(1, total_epochs - warmup_epochs)
            progress = min(1.0, (epoch - warmup_epochs) / span)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(gamma ** ((epoch - warmup_epochs) // max(1, step_size)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device,
                    freeze_backbone=False, use_amp=True):
    model.train()
    if freeze_backbone:
        set_backbone_bn_eval(model)

    running_loss = running_acc = 0.0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        total += bs
        pbar.set_postfix(loss=f"{running_loss/total:.4f}",
                         acc=f"{running_acc/total:.4f}")

    return running_loss / total, running_acc / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    running_loss = running_acc = 0.0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(outputs, labels) * bs
        total += bs

    return running_loss / total, running_acc / total


def train():
    torch.manual_seed(config.SEED)
    if config.NUM_WORKERS == 0:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.AMP and device.type == "cuda"

    print(f"Stage : {config.ACTIVE_STAGE}")
    print(f"Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})"
                                 if device.type == "cuda" else ""))

    banner("Configuration")
    print(f"  Data dir     : {config.DATA_DIR}")
    print(f"  Dataset      : {config.DATASET_ROOT}")
    print(f"  Architecture : {config.ARCHITECTURE}"
          + ("  (ImageNet weights)" if config.PRETRAINED else ""))
    if config.INIT_CHECKPOINT:
        print(f"  Warm start   : {config.INIT_CHECKPOINT}")
    print(f"  Image size   : {config.IMG_SIZE}")
    print(f"  Batch size   : {config.BATCH_SIZE}")
    print(f"  Epochs       : {config.NUM_EPOCHS}")
    print(f"  Head LR      : {config.HEAD_LR}")
    print(f"  Backbone LR  : {config.BACKBONE_LR}")
    print(f"  Freeze epochs: {config.FREEZE_BACKBONE_EPOCHS}")
    print(f"  AMP          : {use_amp}")

    if not os.path.isdir(config.DATASET_ROOT):
        raise SystemExit(
            f"\nDataset not found: {config.DATASET_ROOT}\n"
            f"Set the root with --data-dir or PLANT_DATA_DIR, or clone:\n"
            f"  git clone https://github.com/spMohanty/PlantVillage-Dataset\n"
            f"  git clone https://github.com/pratikkayal/PlantDoc-Dataset\n"
            f"into {config.DATA_DIR}")

    # ── Data ──
    train_loader, val_loader, meta = data.get_dataloaders()
    num_classes = meta["num_classes"]
    idx_to_class = meta["idx_to_class"]

    with open(os.path.join(config.SAVE_DIR, "class_names.json"), "w") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)

    # ── Model ──
    model = build_model(
        num_classes=num_classes,
        architecture=config.ARCHITECTURE,
        pretrained=config.PRETRAINED,
        init_weights=not bool(config.INIT_CHECKPOINT),
        dropout=config.DROPOUT,
        init_from=config.INIT_CHECKPOINT,
    ).to(device)

    frozen = config.FREEZE_BACKBONE_EPOCHS > 0
    if frozen:
        set_backbone_trainable(model, False)
        print(f"Backbone frozen for {config.FREEZE_BACKBONE_EPOCHS} epoch(s).")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters(model), "lr": config.BACKBONE_LR},
            {"params": head_parameters(model), "lr": config.HEAD_LR},
        ],
        lr=config.HEAD_LR, weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = make_scheduler(optimizer, config.NUM_EPOCHS,
                               config.WARMUP_EPOCHS, config.SCHEDULER,
                               config.STEP_LR_SIZE, config.STEP_LR_GAMMA)
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # ── Logging ──
    log_path = os.path.join(config.SAVE_DIR, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc",
                         "val_loss", "val_acc", "lr"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss, best_val_acc, best_epoch, patience = float("inf"), 0.0, 0, 0
    ckpt_path = os.path.join(config.SAVE_DIR, "best_model.pth")

    banner(f"Training {config.ACTIVE_STAGE}  |  {config.ARCHITECTURE}  |  "
           f"{num_classes} classes  |  {config.NUM_EPOCHS} epochs")
    start = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        lr = optimizer.param_groups[1]["lr"]

        if frozen and epoch > config.FREEZE_BACKBONE_EPOCHS:
            frozen = False
            set_backbone_trainable(model, True)
            print(f"  >> Unfroze backbone at epoch {epoch} "
                  f"(backbone LR {optimizer.param_groups[0]['lr']:.2e})")

        print(f"Epoch {epoch}/{config.NUM_EPOCHS}  (head lr={lr:.2e}"
              f"{', frozen backbone' if frozen else ''})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            freeze_backbone=frozen, use_amp=use_amp)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device,
                                     use_amp=use_amp)
        scheduler.step()

        log_writer.writerow([epoch, f"{train_loss:.5f}", f"{train_acc:.5f}",
                             f"{val_loss:.5f}", f"{val_acc:.5f}", f"{lr:.6f}"])
        log_file.flush()
        for key, value in (("train_loss", train_loss), ("train_acc", train_acc),
                           ("val_loss", val_loss), ("val_acc", val_acc)):
            history[key].append(value)

        print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc*100:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, best_epoch = val_loss, val_acc, epoch
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "num_classes": num_classes,
                "idx_to_class": idx_to_class,
                "architecture": config.ARCHITECTURE,
                "stage": config.ACTIVE_STAGE,
                "img_size": config.IMG_SIZE,
                "layout": meta["layout"],
                "uses_official_split": meta["uses_official_split"],
            }, ckpt_path)
            print(f"  [best] saved (val_loss={val_loss:.4f})")
        else:
            patience += 1
            print(f"  No improvement ({patience}/{config.EARLY_STOP_PATIENCE})")

        if patience >= config.EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}.")
            break
        print()

    log_file.close()
    elapsed = time.time() - start

    print("\nLoading best checkpoint for final evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    val_loss, val_acc = evaluate(model, val_loader, criterion, device,
                                 use_amp=use_amp)

    summary = {
        "stage": config.ACTIVE_STAGE,
        "architecture": config.ARCHITECTURE,
        "pretrained": config.PRETRAINED,
        "warm_start": config.INIT_CHECKPOINT,
        "dataset": config.DATASET_ROOT,
        "layout": meta["layout"],
        "uses_official_split": meta["uses_official_split"],
        "metric_split": "validation",
        "num_classes": num_classes,
        "num_train": meta["num_train"],
        "num_val": meta["num_val"],
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "elapsed_min": elapsed / 60.0,
        "img_size": config.IMG_SIZE,
        "batch_size": config.BATCH_SIZE,
        "head_lr": config.HEAD_LR,
        "backbone_lr": config.BACKBONE_LR,
        "freeze_backbone_epochs": config.FREEZE_BACKBONE_EPOCHS,
        "scheduler": config.SCHEDULER,
        "amp": use_amp,
    }
    with open(os.path.join(config.SAVE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(config.SAVE_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    banner(f"Done in {elapsed/60:.1f} min  |  best epoch {best_epoch}  |  "
           f"val acc {val_acc*100:.2f}%")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Log        : {log_path}")
    print(f"\nNext: python evaluate.py --stage {config.ACTIVE_STAGE}")

    return model


if __name__ == "__main__":
    train()

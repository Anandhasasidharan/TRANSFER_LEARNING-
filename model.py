"""
Models for the 3-stage crop-disease pipeline.

Two families:

  CropDiseaseNet   custom 4-block CNN, trained from random init (stage 1)
  TransferModel    torchvision backbone with a fresh classifier head,
                   for ImageNet / PlantVillage warm starts (stages 2 & 3)

Both expose `backbone_module()` and `head_module()` so the training loop can
freeze, unfreeze and set discriminative learning rates without knowing which
family it is holding.

Note on weight initialisation: CropDiseaseNet applies Kaiming init in
__init__. Callers that intend to load a checkpoint afterwards must pass
init_weights=False, otherwise the freshly-initialised kernels are the ones
that survive if the load silently misses keys.
"""

import os

import torch
import torch.nn as nn

try:
    import torchvision.models as tvm
except ImportError:                                   # pragma: no cover
    tvm = None


SUPPORTED_BACKBONES = (
    "resnet18", "resnet34", "resnet50",
    "efficientnet_b0", "efficientnet_b1",
    "mobilenet_v3_small", "mobilenet_v3_large",
    "vgg16", "densenet121",
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── Convolutional Block ─────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Double-conv block: Conv→BN→ReLU → Conv→BN→Pool."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x))


# ─── Custom CNN ──────────────────────────────────────────────────────────────
class CropDiseaseNet(nn.Module):
    """
    Custom CNN for multi-crop, multi-disease classification.

    Input  : (B, 3, 224, 224)
    Output : (B, num_classes)

    Feature extractor spatial sizes (224 input):
        Block 1 (32 filters):  224 → 112
        Block 2 (64 filters):  112 →  56
        Block 3 (128 filters):  56 →  28
        Block 4 (256 filters):  28 →  14
    Then Global Average Pool → 256-d vector → Dropout → Linear(256→num_classes).
    """

    def __init__(self, num_classes: int, init_weights: bool = True, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )
        if init_weights:
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ── uniform interface with TransferModel ──
    def backbone_module(self):
        return self.features

    def head_module(self):
        return self.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ─── Transfer backbone ───────────────────────────────────────────────────────
def _find_head(module):
    """Locate the final Linear layer. Returns (attribute_name, index, in_features)."""
    for attr in ("fc", "classifier", "head", "last_linear"):
        sub = getattr(module, attr, None)
        if isinstance(sub, nn.Linear):
            return attr, None, sub.in_features
        if isinstance(sub, nn.Sequential):
            for i in range(len(sub) - 1, -1, -1):
                if isinstance(sub[i], nn.Linear):
                    return attr, i, sub[i].in_features
    raise ValueError(f"No classifier head found on {type(module).__name__}")


class TransferModel(nn.Module):
    """A torchvision backbone with the original head replaced by Dropout+Linear."""

    def __init__(self, arch: str, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()
        if tvm is None:
            raise ImportError("torchvision is required for transfer learning")
        arch = arch.lower()
        if arch not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone {arch!r}. "
                             f"Choose from: {', '.join(SUPPORTED_BACKBONES)}")

        weights = "DEFAULT" if pretrained else None
        self.arch = arch
        self.pretrained = pretrained
        self.backbone = tvm.get_model(arch, weights=weights)

        attr, index, in_features = _find_head(self.backbone)
        self._head_ref = (attr, index)
        self.in_features = in_features

        if index is None:
            # Head is a bare Linear -> wrap with dropout
            linear = nn.Linear(in_features, num_classes)
            setattr(self.backbone, attr,
                    nn.Sequential(nn.Dropout(p=dropout), linear))
        else:
            seq = getattr(self.backbone, attr)
            for layer in seq:
                if isinstance(layer, nn.Dropout):
                    layer.p = dropout
            linear = nn.Linear(in_features, num_classes)
            seq[index] = linear

        # Keep a reference in a plain list so nn.Module does not register the
        # same Linear under a second name (which would duplicate state_dict
        # keys and double-count it in model.parameters()).
        self._head_holder = [linear]

        self._dropout = dropout
        self.num_classes = num_classes

    # ── uniform interface with CropDiseaseNet ──
    def backbone_module(self):
        return self.backbone

    def head_module(self):
        return self._head_holder[0]

    def forward(self, x):
        return self.backbone(x)


# ─── Freeze / unfreeze / discriminative LR helpers ───────────────────────────
def head_parameters(model):
    return list(model.head_module().parameters())


def backbone_parameters(model):
    head_ids = {id(p) for p in head_parameters(model)}
    return [p for p in model.parameters() if id(p) not in head_ids]


def set_backbone_trainable(model, trainable: bool):
    """Freeze or unfreeze everything except the classifier head."""
    head_ids = {id(p) for p in head_parameters(model)}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = trainable


def set_backbone_bn_eval(model):
    """Keep frozen BatchNorm layers in eval mode.

    Without this, a frozen backbone keeps updating its running mean/var on the
    new dataset, which silently corrupts the pretrained features.
    """
    for m in model.backbone_module().modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Checkpoint warm start ───────────────────────────────────────────────────
def load_checkpoint_into(model, checkpoint_path: str, device="cpu"):
    """Load weights from a checkpoint, skipping shape mismatches.

    Used for stage 3: the classifier head changes size (38 -> 27 classes), so
    its weights are dropped while the whole backbone transfers.
    """
    if not checkpoint_path:
        return 0, []
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Warm-start checkpoint not found: {checkpoint_path}\n"
            f"Run the earlier stage first, or edit init_from in config.py."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    src_arch = ckpt.get("architecture") if isinstance(ckpt, dict) else None

    target = model.state_dict()
    reusable, skipped = {}, []
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        if key in target and target[key].shape == value.shape:
            reusable[key] = value
        else:
            skipped.append(key)

    missing = [k for k in target if k not in reusable]
    model.load_state_dict(reusable, strict=False)

    print(f"Warm start from {os.path.basename(os.path.dirname(checkpoint_path))}/"
          f"{os.path.basename(checkpoint_path)}")
    if src_arch:
        print(f"  Source architecture : {src_arch}")
    print(f"  Transferred tensors : {len(reusable)}")
    print(f"  Re-initialised      : {len(missing)} "
          f"(head: {sum('classifier' in m or '.fc.' in m for m in missing)})")
    if skipped:
        print(f"  Skipped (shape/name): {len(skipped)}")
    return len(reusable), missing


# ─── Factory ─────────────────────────────────────────────────────────────────
def build_model(num_classes: int, architecture: str = "custom",
                pretrained: bool = False, init_weights: bool = True,
                dropout: float = 0.4, init_from: str = None):
    """Create a model, optionally warm-starting it from a checkpoint."""
    architecture = (architecture or "custom").lower()

    if architecture == "custom":
        model = CropDiseaseNet(num_classes, init_weights=init_weights, dropout=dropout)
        label = "CropDiseaseNet"
    else:
        if not pretrained and not init_from:
            print(f"  Note: {architecture} built with random init "
                  f"(no ImageNet weights, no checkpoint).")
        model = TransferModel(architecture, num_classes,
                              pretrained=pretrained, dropout=dropout)
        label = f"TransferModel({architecture})"

    if init_from:
        load_checkpoint_into(model, init_from)

    total = sum(p.numel() for p in model.parameters())
    trainable = count_parameters(model)
    print(f"{label} created — {num_classes} classes | "
          f"{total:,} params ({trainable:,} trainable)")
    return model


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)

    m = build_model(num_classes=33, architecture="custom")
    print(f"  custom     {tuple(x.shape)} -> {tuple(m(x).shape)}")

    for arch in ("resnet18", "efficientnet_b0", "mobilenet_v3_small"):
        m = build_model(num_classes=27, architecture=arch, pretrained=False)
        print(f"  {arch:<18s} {tuple(x.shape)} -> {tuple(m(x).shape)} | "
              f"head={m.head_module().out_features} in={m.in_features}")
        assert len(backbone_parameters(m)) > 0

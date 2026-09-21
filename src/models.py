"""Model factory covering all 10 required architectures.

    get_segmentation_model(name, num_classes, pretrained=True, **kwargs)

Names (canonical / aliases):
    kmeans                                                              (traditional)
    fcn                     alias: fcn_resnet50                         (semantic)
    unet                                                                (semantic)
    unetpp                  alias: unet++                               (semantic)
    segnet                                                              (semantic)
    deeplabv3               alias: deeplabv3_resnet50                   (semantic)
    pspnet                                                              (semantic)
    segformer               alias: segformer_b0                         (semantic)
    maskrcnn                alias: mask_rcnn                            (instance)
    yolo_seg                alias: yolov8_seg                           (instance)

All deep models return either an nn.Module (semantic) or a ready-to-train
torchvision Mask R-CNN / HuggingFace SegFormer / Ultralytics YOLO wrapper.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Torch imports guarded so `import models` still works for scripts that only need KMeans.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models.segmentation import (
        fcn_resnet50, FCN_ResNet50_Weights,
        deeplabv3_resnet50, DeepLabV3_ResNet50_Weights,
    )
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    _HAS_TORCH = True
except Exception:  # torch not installed - keep module importable for KMeans-only usage
    _HAS_TORCH = False
    class _AttrStub:
        """A permissive stub whose attributes always resolve to a no-op class."""
        def __getattr__(self, _):
            class _C:
                def __init__(self, *a, **kw): pass
                def __call__(self, *a, **kw): raise ImportError("PyTorch not installed")
            return _C
    class _NNStub:
        class Module:  # type: ignore
            def __init__(self, *a, **kw): pass
        def __getattr__(self, _):
            class _C:
                def __init__(self, *a, **kw): pass
                def __call__(self, *a, **kw): raise ImportError("PyTorch not installed")
            return _C
    torch = _AttrStub()      # type: ignore
    nn = _NNStub()           # type: ignore
    F = _AttrStub()          # type: ignore


# ==================== 1. Traditional baseline ==================== #

class KMeansSegmenter:
    """K-Means color segmentation baseline (assignment Model 1).

    Not a class-aware model: it groups pixels by RGB (or Lab) similarity. Per
    the assignment, cluster IDs are NOT class labels; we retain the option to
    fit a class mapping from training data only, but the default forward pass
    returns cluster assignments for qualitative comparison.
    """
    def __init__(self, num_classes: int = 6, n_clusters: Optional[int] = None,
                 color_space: str = "lab"):
        self.num_classes = num_classes
        self.n_clusters = n_clusters or num_classes
        self.color_space = color_space
        self.class_map = None  # cluster_id -> class_id (fit on training data)

    def _to_features(self, img_rgb):
        import numpy as np
        from skimage import color
        if self.color_space == "lab":
            feats = color.rgb2lab(img_rgb / 255.0)
        else:
            feats = img_rgb.astype(float) / 255.0
        H, W, C = feats.shape
        return feats.reshape(-1, C), (H, W)

    def predict(self, img_rgb):
        """Return cluster IDs mapped through class_map (if fitted), else clusters."""
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np
        X, (H, W) = self._to_features(img_rgb)
        km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, n_init=3, batch_size=1024)
        clusters = km.fit_predict(X).reshape(H, W)
        if self.class_map is not None:
            out = np.zeros_like(clusters)
            for c, cls in self.class_map.items():
                out[clusters == c] = cls
            return out
        return clusters

    def fit_class_mapping(self, images, masks):
        """Assign each cluster to the majority class it covers across training data.

        The assignment forbids fitting to test labels; this fits only to training.
        """
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans
        Xs = []
        Ys = []
        for img, mask in zip(images, masks):
            X, _ = self._to_features(img)
            Y = mask.reshape(-1)
            keep = Y != 255
            Xs.append(X[keep]); Ys.append(Y[keep])
        X = np.concatenate(Xs); Y = np.concatenate(Ys)
        km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42, n_init=3, batch_size=4096)
        c = km.fit_predict(X)
        mapping = {}
        for k in range(self.n_clusters):
            in_c = Y[c == k]
            mapping[k] = int(np.bincount(in_c, minlength=self.num_classes).argmax()) if len(in_c) else 0
        self._km_global = km
        self.class_map = mapping
        return self


# ==================== 2. FCN-ResNet50 ==================== #

def build_fcn(num_classes: int, pretrained: bool = True):
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for FCN")
    weights = FCN_ResNet50_Weights.DEFAULT if pretrained else None
    model = fcn_resnet50(weights=weights, num_classes=21 if pretrained else num_classes,
                         aux_loss=True)
    if pretrained:
        # Replace classifier heads for our class count
        model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ==================== 3. U-Net ==================== #

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 6, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = DoubleConv(3,  c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.bott = DoubleConv(c4, c4 * 2)
        self.pool = nn.MaxPool2d(2)

        self.up4  = nn.ConvTranspose2d(c4 * 2, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4 * 2, c4)
        self.up3  = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3 * 2, c3)
        self.up2  = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2 * 2, c2)
        self.up1  = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1 * 2, c1)
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bott(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return {"out": self.head(d1)}


# ==================== 4. U-Net++ ==================== #

class UNetPlusPlus(nn.Module):
    """Nested U-Net with dense skip pathways (Zhou et al. 2018)."""
    def __init__(self, num_classes: int = 6, channels=(64, 128, 256, 512),
                 deep_supervision: bool = False):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Backbone (column 0)
        self.x00 = DoubleConv(3,  c1)
        self.x10 = DoubleConv(c1, c2)
        self.x20 = DoubleConv(c2, c3)
        self.x30 = DoubleConv(c3, c4)

        # Nested column 1
        self.x01 = DoubleConv(c1 + c2, c1)
        self.x11 = DoubleConv(c2 + c3, c2)
        self.x21 = DoubleConv(c3 + c4, c3)

        # Nested column 2
        self.x02 = DoubleConv(c1 * 2 + c2, c1)
        self.x12 = DoubleConv(c2 * 2 + c3, c2)

        # Nested column 3
        self.x03 = DoubleConv(c1 * 3 + c2, c1)

        self.deep_supervision = deep_supervision
        self.head = nn.Conv2d(c1, num_classes, 1)
        if deep_supervision:
            self.head1 = nn.Conv2d(c1, num_classes, 1)
            self.head2 = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))

        x01 = self.x01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.x11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.x21(torch.cat([x20, self.up(x30)], 1))

        x02 = self.x02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.x12(torch.cat([x10, x11, self.up(x21)], 1))

        x03 = self.x03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        if self.deep_supervision and self.training:
            return {"out": self.head(x03),
                    "aux1": self.head1(x01),
                    "aux2": self.head2(x02)}
        return {"out": self.head(x03)}


# ==================== 5. SegNet ==================== #

class SegNet(nn.Module):
    """SegNet with pooling-index-based upsampling (Badrinarayanan et al. 2017)."""
    def __init__(self, num_classes: int = 6):
        super().__init__()
        def enc(in_c, out_c, n):
            layers = []
            for i in range(n):
                layers += [nn.Conv2d(in_c if i == 0 else out_c, out_c, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)
        self.enc1 = enc(3,   64, 2)
        self.enc2 = enc(64, 128, 2)
        self.enc3 = enc(128, 256, 3)
        self.enc4 = enc(256, 512, 3)
        self.enc5 = enc(512, 512, 3)

        self.dec5 = enc(512, 512, 3)
        self.dec4 = enc(512, 256, 3)
        self.dec3 = enc(256, 128, 3)
        self.dec2 = enc(128,  64, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, padding=1),
        )
        self.pool   = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        e1 = self.enc1(x); p1, i1 = self.pool(e1)
        e2 = self.enc2(p1); p2, i2 = self.pool(e2)
        e3 = self.enc3(p2); p3, i3 = self.pool(e3)
        e4 = self.enc4(p3); p4, i4 = self.pool(e4)
        e5 = self.enc5(p4); p5, i5 = self.pool(e5)

        d5 = self.dec5(self.unpool(p5, i5, output_size=e5.size()))
        d4 = self.dec4(self.unpool(d5, i4, output_size=e4.size()))
        d3 = self.dec3(self.unpool(d4, i3, output_size=e3.size()))
        d2 = self.dec2(self.unpool(d3, i2, output_size=e2.size()))
        d1 = self.dec1(self.unpool(d2, i1, output_size=e1.size()))
        return {"out": d1}


# ==================== 6. DeepLabV3-ResNet50 ==================== #

def build_deeplabv3(num_classes: int, pretrained: bool = True):
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for DeepLabV3")
    weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(weights=weights, num_classes=21 if pretrained else num_classes,
                               aux_loss=True)
    if pretrained:
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
        if model.aux_classifier is not None:
            model.aux_classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


# ==================== 7. PSPNet ==================== #

class PPM(nn.Module):
    """Pyramid Pooling Module."""
    def __init__(self, in_c: int, bins=(1, 2, 3, 6)):
        super().__init__()
        red_c = in_c // len(bins)
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_c, red_c, 1, bias=False),
                nn.BatchNorm2d(red_c), nn.ReLU(inplace=True),
            ) for b in bins
        ])
        self.out_channels = in_c + red_c * len(bins)

    def forward(self, x):
        h, w = x.shape[2:]
        pooled = [x]
        for f in self.features:
            y = f(x)
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            pooled.append(y)
        return torch.cat(pooled, dim=1)


class PSPNet(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.backbone = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.ppm = PPM(2048, bins=(1, 2, 3, 6))
        self.head = nn.Sequential(
            nn.Conv2d(self.ppm.out_channels, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, 1),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        f = self.backbone(x)
        f = self.ppm(f)
        out = self.head(f)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return {"out": out}


# ==================== 8. Mask R-CNN ==================== #

def build_maskrcnn(num_classes: int, pretrained: bool = True):
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for Mask R-CNN")
    # +1 for background class as required by torchvision detection heads
    n = num_classes  # our num_classes already includes background at index 0
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, n)
    hidden = 256
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, hidden, n)
    return model


# ==================== 9. YOLO Segmentation ==================== #

def build_yolo_seg(variant: str = "yolov8n-seg.pt"):
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Install ultralytics: pip install ultralytics") from e
    return YOLO(variant)


# ==================== 10. SegFormer ==================== #

def build_segformer(num_classes: int, variant: str = "nvidia/mit-b0", pretrained: bool = True):
    try:
        from transformers import SegformerForSemanticSegmentation  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Install transformers: pip install transformers") from e
    return SegformerForSemanticSegmentation.from_pretrained(
        variant,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )


# ==================== Factory ==================== #

_MODEL_ALIASES = {
    "fcn_resnet50": "fcn",
    "unet++": "unetpp",
    "unetplusplus": "unetpp",
    "deeplabv3_resnet50": "deeplabv3",
    "deeplab": "deeplabv3",
    "segformer_b0": "segformer",
    "mask_rcnn": "maskrcnn",
    "yolov8_seg": "yolo_seg",
    "yolo": "yolo_seg",
}


def get_segmentation_model(model_name: str, num_classes: int = 6, pretrained: bool = True,
                           **kwargs) -> Any:
    """Uniform factory used by both benchmark and per-model scripts."""
    name = _MODEL_ALIASES.get(model_name.lower().strip(), model_name.lower().strip())
    if name == "kmeans":
        return KMeansSegmenter(num_classes=num_classes, **kwargs)
    if name == "fcn":
        return build_fcn(num_classes, pretrained=pretrained)
    if name == "unet":
        return UNet(num_classes=num_classes)
    if name == "unetpp":
        return UNetPlusPlus(num_classes=num_classes,
                            deep_supervision=kwargs.get("deep_supervision", False))
    if name == "segnet":
        return SegNet(num_classes=num_classes)
    if name == "deeplabv3":
        return build_deeplabv3(num_classes, pretrained=pretrained)
    if name == "pspnet":
        return PSPNet(num_classes=num_classes, pretrained=pretrained)
    if name == "segformer":
        return build_segformer(num_classes, variant=kwargs.get("variant", "nvidia/mit-b0"),
                                pretrained=pretrained)
    if name == "maskrcnn":
        return build_maskrcnn(num_classes, pretrained=pretrained)
    if name == "yolo_seg":
        return build_yolo_seg(variant=kwargs.get("variant", "yolov8n-seg.pt"))
    raise ValueError(f"Unknown model name: {model_name}")


def count_parameters(model) -> Dict[str, int]:
    """Return {total, trainable} parameter counts."""
    if not hasattr(model, "parameters"):
        return {"total": 0, "trainable": 0}
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(train)}

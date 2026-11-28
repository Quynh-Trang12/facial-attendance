"""Capture/stress transforms and exact device colour augmentation.

Device colour tables come from installed Pillow uint8 conversions, preserving
its RGB/HSV rounding rather than approximating it. Per-image jitter order and
factors use the seeded torchvision recipe; model computation remains FP32.
"""

from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import (
    ColorJitter,
    Compose,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

from facial_attendance.liveness.models.binary_model import ARCHITECTURE, face_transform


def jpeg_image(image, quality):
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


def downsample_image(image, short_side):
    scale = min(1, short_side / min(image.size))
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


class CaptureAugmentation:
    """Random crop/flip/color from the papers, plus camera degradation probes.

    Numeric ranges are project settings, not a claimed optimal paper recipe.
    Both classes receive the same transformations to avoid camera=spoof cues.
    """

    def __init__(self, architecture=ARCHITECTURE):
        self.crop = RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.65, 0.95))
        self.flip = RandomHorizontalFlip()
        self.color = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        self.tensor = face_transform(architecture=architecture)

    def __call__(self, image):
        image = self.flip(self.crop(image))
        if torch.rand(()).item() < 0.8:
            image = self.color(image)
        if torch.rand(()).item() < 0.5:
            image = downsample_image(image, int(torch.randint(64, 161, ()).item()))
        if torch.rand(()).item() < 0.35:
            image = jpeg_image(image, int(torch.randint(50, 96, ()).item()))
        return self.tensor(image)


class CaptureStressView:
    """Fixed validation probe; correlated views are not independent samples."""

    def __init__(self, architecture=ARCHITECTURE):
        self.tensor = face_transform(architecture=architecture)

    def __call__(self, image):
        image = downsample_image(image, 64)
        values = np.clip(np.asarray(image, dtype=np.float32) * 1.5 + 16, 0, 255).astype(
            np.uint8
        )
        return self.tensor(jpeg_image(Image.fromarray(values), 70))


_tables = {}


class DeviceColorParameters:
    def __init__(self, *, brightness, contrast, saturation, hue):
        self.jitter = ColorJitter(brightness, contrast, saturation, hue)
        self.resize = Resize((224, 224))

    def __call__(self, image, targets):
        if image.mode != "RGB":
            raise ValueError("Colour augmentation requires RGB")
        image = self.resize(image)
        jitter = self.jitter
        order, brightness, contrast, saturation, hue = ColorJitter.get_params(
            jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue
        )
        enabled = torch.tensor(
            [value is not None for value in (brightness, contrast, saturation, hue)]
        )
        order[~enabled[order]] = -1
        factors = torch.tensor(
            [
                1.0 if value is None else value
                for value in (brightness, contrast, saturation)
            ],
            dtype=torch.float32,
        )
        # torchvision's PIL path truncates hue*255, then wraps modulo 256.
        shift = int((0.0 if hue is None else hue) * 255) % 256
        pixels = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(
            2, 0, 1
        )
        return pixels, {
            **targets,
            "_color_order": order,
            "_color_factors": factors,
            "_color_hue": torch.tensor(shift, dtype=torch.int64),
        }


def _conversion_tables(device):
    key = str(device)
    if key not in _tables:
        values = np.arange(1 << 24, dtype=np.uint32)
        pixels = np.stack(
            (values >> 16, (values >> 8) & 255, values & 255), axis=1
        ).astype(np.uint8)
        pixels = pixels.reshape(4096, 4096, 3)
        rgb = Image.fromarray(pixels, "RGB")
        hsv = torch.from_numpy(
            np.array(rgb.convert("HSV"), copy=True).reshape(-1, 3)
        ).to(device)
        luma = torch.from_numpy(np.array(rgb.convert("L"), copy=True).reshape(-1)).to(
            device
        )
        inverse = torch.from_numpy(
            np.array(Image.fromarray(pixels, "HSV").convert("RGB"), copy=True).reshape(
                -1, 3
            )
        ).to(device)
        # This CPU lookup preserves ToTensor's exact FP32 division values even
        # when the CUDA division implementation rounds differently.
        unit = torch.arange(256, dtype=torch.float32).div(255).to(device)
        _tables[key] = hsv, luma, inverse, unit
    return _tables[key]


def _indices(images):
    red, green, blue = images.unbind(1)
    return (
        (red.to(torch.int32) << 16)
        | (green.to(torch.int32) << 8)
        | blue.to(torch.int32)
    )


def _lookup_rgb(table, indices):
    return (
        table.index_select(0, indices.reshape(-1))
        .reshape(*indices.shape, 3)
        .permute(0, 3, 1, 2)
    )


def _blend(images, base, factor):
    # Separate multiply/add match Pillow's float32 blend and uint8 truncation;
    # using lerp/FMA or float HSV conversions would change some pixel values.
    return (
        ((images.float() - base) * factor[:, None, None, None] + base)
        .clamp(0, 255)
        .to(torch.uint8)
    )


@torch.no_grad()
def apply_device_color(images, order, factors, hue):
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected NCHW uint8 RGB images")
    if order.device.type != "cpu" or order.shape != (len(images), 4):
        raise ValueError("Expected the CPU per-image colour order")
    hsv_table, luma_table, rgb_table, unit_table = _conversion_tables(images.device)
    images = images.clone()
    for position in range(4):
        for operation in range(4):
            selected = (order[:, position] == operation).nonzero().flatten()
            if not len(selected):
                continue
            selected = selected.to(images.device, non_blocking=True)
            current = images.index_select(0, selected)
            if operation == 0:
                result = _blend(current, 0.0, factors.index_select(0, selected)[:, 0])
            elif operation in (1, 2):
                luminance = luma_table.index_select(
                    0, _indices(current).reshape(-1)
                ).reshape(len(current), *current.shape[-2:])
                if operation == 1:
                    pixels = current.shape[-2] * current.shape[-1]
                    base = (
                        (luminance.sum((1, 2), dtype=torch.int64) + pixels // 2)
                        // pixels
                    ).float()[:, None, None, None]
                else:
                    base = luminance.float()[:, None]
                result = _blend(
                    current, base, factors.index_select(0, selected)[:, operation]
                )
            else:
                hsv = _lookup_rgb(hsv_table, _indices(current))
                shifted = (
                    hsv[:, 0].to(torch.int32)
                    + hue.index_select(0, selected)[:, None, None]
                ).bitwise_and(255)
                indices = (
                    (shifted.to(torch.int32) << 16)
                    | (hsv[:, 1].to(torch.int32) << 8)
                    | hsv[:, 2].to(torch.int32)
                )
                result = _lookup_rgb(rgb_table, indices)
            images.index_copy_(0, selected, result)
    return unit_table.index_select(0, images.reshape(-1).to(torch.int32)).reshape(
        images.shape
    )


class GeometricColorAugmentation:
    """Color-only RGB augmentation preserves teacher-map crop coordinates.

    The paper omits color-distortion strengths; the caller supplies them rather
    than presenting arbitrary defaults as published hyperparameters.
    """

    def __init__(self, *, brightness, contrast, saturation, hue):
        self.transform = Compose(
            [
                Resize((224, 224)),
                ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
                ToTensor(),
            ]
        )

    def __call__(self, image, targets):
        if image.mode != "RGB":
            raise ValueError("Geometric training input must be RGB")
        for name, channels in (("depth_map", 1), ("reflection_map", 3)):
            if targets[name].shape != (channels, 14, 14):
                raise ValueError("Geometric maps must be aligned 14x14 targets")
        return self.transform(image), targets

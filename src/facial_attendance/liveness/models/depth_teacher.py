"""PRNet resfcn256 inference with source TensorFlow SAME padding semantics.

Architecture: yfeng95/PRNet/predictor.py. Frozen batch statistics, UV position
coordinates and MaxPos=256*1.1 follow the original inference implementation.The supplied-box crop, restored mesh and mean-triangle visibility produce a
14 by 14 relative-depth target. That grid is a project choice; its values are
normalized within the face, not measured distances in metres.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrozenSourceBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        for name, value in (
            ("gamma", torch.ones(channels)),
            ("beta", torch.zeros(channels)),
            ("mean", torch.zeros(channels)),
            ("variance", torch.ones(channels)),
        ):
            self.register_buffer(name, value)

    def forward(self, images):
        return F.batch_norm(
            images,
            self.mean,
            self.variance,
            self.gamma,
            self.beta,
            training=False,
            eps=0.001,
        )


class SourceConv(nn.Module):
    def __init__(
        self,
        channels,
        outputs,
        kernel,
        stride=1,
        *,
        normalized=True,
        transpose=False,
        activation="relu",
    ):
        super().__init__()
        self.kernel, self.stride, self.transpose, self.activation = (
            kernel,
            stride,
            transpose,
            activation,
        )
        cls = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = cls(channels, outputs, kernel, stride=stride, bias=False)
        self.norm = FrozenSourceBatchNorm(outputs) if normalized else None

    def forward(self, images):
        if self.transpose:
            # TF SAME transpose returns input*stride, including even kernels.
            value = self.conv(images)
            start = (self.kernel - self.stride) // 2
            height, width = (
                images.shape[-2] * self.stride,
                images.shape[-1] * self.stride,
            )
            value = value[:, :, start : start + height, start : start + width]
        else:
            pads = []
            for size in reversed(images.shape[-2:]):
                total = max(
                    (math.ceil(size / self.stride) - 1) * self.stride
                    + self.kernel
                    - size,
                    0,
                )
                pads.extend((total // 2, total - total // 2))
            value = self.conv(F.pad(images, pads))
        if self.norm is not None:
            value = self.norm(value)
        if self.activation == "relu":
            return value.relu()
        if self.activation == "sigmoid":
            return value.sigmoid()
        return value


class SourceResidual(nn.Module):
    def __init__(self, channels, outputs, stride):
        super().__init__()
        self.shortcut = (
            SourceConv(channels, outputs, 1, stride, normalized=False, activation=None)
            if stride != 1 or channels != outputs
            else None
        )
        self.first = SourceConv(channels, outputs // 2, 1)
        self.second = SourceConv(outputs // 2, outputs // 2, 4, stride)
        self.third = SourceConv(
            outputs // 2, outputs, 1, normalized=False, activation=None
        )
        self.norm = FrozenSourceBatchNorm(outputs)

    def forward(self, images):
        shortcut = images if self.shortcut is None else self.shortcut(images)
        return self.norm(self.third(self.second(self.first(images))) + shortcut).relu()


class PRNetDepthTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = SourceConv(3, 16, 4)
        self.encoder = nn.ModuleList()
        channels = 16
        for outputs in (32, 64, 128, 256, 512):
            self.encoder.extend(
                (
                    SourceResidual(channels, outputs, 2),
                    SourceResidual(outputs, outputs, 1),
                )
            )
            channels = outputs
        self.decoder = nn.ModuleList()
        specifications = (
            (512, 1),
            (256, 2),
            (256, 1),
            (256, 1),
            (128, 2),
            (128, 1),
            (128, 1),
            (64, 2),
            (64, 1),
            (64, 1),
            (32, 2),
            (32, 1),
            (16, 2),
            (16, 1),
            (3, 1),
            (3, 1),
            (3, 1),
        )
        for index, (outputs, stride) in enumerate(specifications):
            self.decoder.append(
                SourceConv(
                    channels,
                    outputs,
                    4,
                    stride,
                    transpose=True,
                    activation="sigmoid"
                    if index == len(specifications) - 1
                    else "relu",
                )
            )
            channels = outputs

    def forward(self, rgb):
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 256, 256):
            raise ValueError("PRNet requires NCHW RGB 256x256 input in [0,1]")
        value = self.initial(rgb)
        for block in self.encoder:
            value = block(value)
        for layer in self.decoder:
            value = layer(value)
        return value * (256 * 1.1)


def prnet_face_input(rgb):
    if rgb.ndim != 3 or rgb.shape[0] != 3 or min(rgb.shape[1:]) < 2:
        raise ValueError("Expected CHW RGB face crop with both dimensions >=2")
    height, width = rgb.shape[-2:]
    size = int((width + height) / 2 * 1.6)
    left, top = width / 2 - size / 2, height / 2 - size / 2
    scale = 255 / size
    coordinates = torch.arange(256, dtype=torch.float32, device=rgb.device) / scale
    y, x = torch.meshgrid(coordinates + top, coordinates + left, indexing="ij")
    grid = torch.stack((2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1), dim=-1)
    cropped = F.grid_sample(
        rgb.unsqueeze(0),
        grid.unsqueeze(0),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return cropped, (left, top, scale, width, height)


def restore_prnet_vertices(position, face_indices, alignment):
    """Map PRNet mesh vertices back to the face crop's 14 by 14 target grid.

    Return x/y coordinates in target-grid units and per-face normalized z in [0, 1].
    """
    if position.shape != (3, 256, 256):
        raise ValueError("Expected PRNet 3x256x256 UV position map")
    left, top, scale, width, height = alignment
    vertices = position.flatten(1).T[face_indices].clone()
    vertices[:, 0] = (vertices[:, 0] / scale + left) * 13 / (width - 1)
    vertices[:, 1] = (vertices[:, 1] / scale + top) * 13 / (height - 1)
    vertices[:, 2] /= scale
    depth = vertices[:, 2]
    span = depth.max() - depth.min()
    if not torch.isfinite(vertices).all() or span <= 0:
        raise ValueError("Invalid PRNet mesh or degenerate depth range")
    vertices[:, 2] = (depth - depth.min()) / span
    return vertices


def rasterize_depth(vertices, triangles):
    """Fill the 14 by 14 grid using the mean vertex depth of visible triangles.

    For each grid position covered by multiple triangles, keep the largest mean
    depth, matching the reference renderer's visibility rule. Uncovered cells are zero.
    """
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or triangles.dtype != torch.int64
        or not torch.isfinite(vertices).all()
        or triangles.min() < 0
        or triangles.max() >= len(vertices)
    ):
        raise ValueError("Expected finite vertices and valid indexed triangles")
    mesh = vertices[triangles]
    y, x = torch.meshgrid(
        torch.arange(14, device=vertices.device),
        torch.arange(14, device=vertices.device),
        indexing="ij",
    )
    points = torch.stack((x.flatten(), y.flatten()), dim=1).to(vertices.dtype)
    first = mesh[:, 0, :2]
    v0 = mesh[:, 2, :2] - first
    v1 = mesh[:, 1, :2] - first
    v2 = points.unsqueeze(0) - first.unsqueeze(1)
    dot00 = (v0 * v0).sum(1).unsqueeze(1)
    dot01 = (v0 * v1).sum(1).unsqueeze(1)
    dot11 = (v1 * v1).sum(1).unsqueeze(1)
    dot02 = (v2 * v0.unsqueeze(1)).sum(2)
    dot12 = (v2 * v1.unsqueeze(1)).sum(2)
    denominator = dot00 * dot11 - dot01.square()
    safe = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
    u = (dot11 * dot02 - dot01 * dot12) / safe
    v = (dot00 * dot12 - dot01 * dot02) / safe
    inside = (denominator != 0) & (u >= 0) & (v >= 0) & (u + v < 1)
    triangle_depth = mesh[:, :, 2].mean(1).unsqueeze(1)
    depth = torch.where(inside, triangle_depth, float("-inf")).amax(0)
    return torch.where(torch.isfinite(depth), depth, 0.0).reshape(1, 14, 14)

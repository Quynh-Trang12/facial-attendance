"""Predict live/spoof, semantic labels and geometric targets from shared features.

Reference: Zhang et al., CelebA-Spoof, section 4 and Figure 5:
https://arxiv.org/abs/2007.12342v3. The source_types argument sets the semantic
source-class outputs; the full-source comparison supplies all eleven class IDs.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from facial_attendance.liveness.models.binary_model import ResNetBinaryClassifier

SOURCE_TYPES = (0, 1, 2, 3, 7, 8, 9)
ALL_SOURCE_TYPES = tuple(range(11))
SEMANTIC_TASKS = ("face_attributes", "spoof_type", "illumination")
GEOMETRIC_TASKS = ("depth_map", "reflection_map")


def active_auxiliary_tasks(*, auxiliary=True, geometry=False, tasks=None):
    """Validate enabled heads and return them in stable semantic/geometric loss order."""
    allowed = SEMANTIC_TASKS + (GEOMETRIC_TASKS if geometry else ())
    selected = tuple(
        allowed if tasks is None and auxiliary else () if tasks is None else tasks
    )
    if (
        len(set(selected)) != len(selected)
        or set(selected) - set(allowed)
        or (not auxiliary and selected)
    ):
        raise ValueError("Invalid or disabled auxiliary supervision tasks")
    return tuple(name for name in allowed if name in selected)


@dataclass(frozen=True)
class AuxiliaryLossWeights:
    """Task-loss multipliers; the binary classification loss is included separately."""

    face_attributes: float = 1.0
    spoof_type: float = 0.1
    illumination: float = 0.01
    depth_map: float = 0.1
    reflection_map: float = 0.1


def annotation_targets(attributes, *, source_types=SOURCE_TYPES):
    """Validate the original 44 fields, including no-attribute spoof targets."""
    if len(attributes) != 44 or any(type(value) is not int for value in attributes):
        raise ValueError("Expected 44 integer source attributes")
    source_type, illumination, binary = attributes[40], attributes[41], attributes[43]
    face_attributes = attributes[:40]
    if source_type not in source_types or binary != int(source_type != 0):
        raise ValueError("Invalid project spoof type or conflicting binary label")
    if not 0 <= illumination <= 4 or (illumination == 0) != (binary == 0):
        raise ValueError("Invalid source illumination/no-illumination label")
    if any(value not in (0, 1) for value in face_attributes):
        raise ValueError("Face attributes must be binary")
    # Figure 5 supervises low face-attribute scores for spoof images; masking
    # their loss out would change the method into live-only attribute learning.
    if binary and any(face_attributes):
        raise ValueError("Spoof images must have the source no-attribute vector")
    return {
        "binary": binary,
        "spoof_type": source_types.index(source_type),
        "illumination": illumination,
        "face_attributes": torch.tensor(face_attributes, dtype=torch.float32),
    }


def deterministic_geometry_resize(images):
    """Bilinear align_corners=False via separable matrices.

    CUDA interpolate backward can use nondeterministic atomic accumulation.
    Fixed resize matrices preserve its coordinates while allowing deterministic
    matrix multiplication and reproducible epoch-boundary checkpoints.
    """
    matrices = []
    for length in images.shape[-2:]:
        coordinate = (
            (torch.arange(14, device=images.device, dtype=images.dtype) + 0.5)
            * length
            / 14
            - 0.5
        ).clamp(0, length - 1)
        lower = coordinate.floor().long()
        upper = (lower + 1).clamp(max=length - 1)
        weight = coordinate - lower
        matrix = F.one_hot(lower, length).to(images.dtype) * (1 - weight).unsqueeze(
            1
        ) + F.one_hot(upper, length).to(images.dtype) * weight.unsqueeze(1)
        matrices.append(matrix)
    return torch.matmul(torch.matmul(matrices[0], images), matrices[1].T)


class MultitaskLivenessModel(nn.Module):
    """Use the ResNet-18 features for binary and selected auxiliary predictions.

    The binary output classifies live/spoof. Semantic heads predict face attributes,
    source type and illumination; geometric heads predict depth and RGB reflection
    maps. The caller selects which auxiliary predictions contribute training losses.
    """

    def __init__(self, binary_model, *, source_types=SOURCE_TYPES, geometry=False):
        super().__init__()
        self.binary_model = binary_model
        self.is_resnet = isinstance(binary_model, ResNetBinaryClassifier)
        if source_types not in (SOURCE_TYPES, ALL_SOURCE_TYPES):
            raise ValueError(
                "Use project print/screen types or all eleven source types"
            )
        if geometry and not self.is_resnet:
            raise ValueError(
                "AENet geometric supervision requires the ResNet spatial embedding"
            )
        self.geometry = geometry
        features = (
            binary_model.backbone.fc.in_features
            if self.is_resnet
            else binary_model.classifier[-1].in_features
        )
        self.face_attributes = nn.Linear(features, 40)
        self.spoof_type = nn.Linear(features, len(source_types))
        self.illumination = nn.Linear(features, 5)
        if geometry:
            # Match the author's released depth_final/reflect_final: 512-channel
            # embedding, bias-free 3x3 heads, sigmoid then 14x14 bilinear resize.
            # Preserve the 7x7 spatial grid before upsampling, as in the
            # author's released model, so each target pixel refers to that grid.
            self.depth_map = nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False)
            self.reflection_map = nn.Conv2d(
                512, 3, kernel_size=3, padding=1, bias=False
            )
            for head in (self.depth_map, self.reflection_map):
                # Match the initialization loop in the author's models.py.
                nn.init.normal_(
                    head.weight, mean=0.0, std=math.sqrt(2 / (9 * head.out_channels))
                )

    def forward_tasks(self, images, *, auxiliary=True, tasks=None):
        """Return binary logits and only the requested semantic/geometric outputs.

        Disabled heads are not executed, so they cannot consume RNG or receive gradients.
        """
        selected = active_auxiliary_tasks(
            auxiliary=auxiliary, geometry=self.geometry, tasks=tasks
        )
        if self.is_resnet:
            spatial = self.binary_model.feature_map(images)
            embedding = self.binary_model.backbone.avgpool(spatial).flatten(1)
            binary_classes = self.binary_model.backbone.fc(embedding)
            binary = binary_classes[:, 1] - binary_classes[:, 0]
        else:
            features = self.binary_model.features(images)
            embedding = self.binary_model.avgpool(features).flatten(1)
            for layer in list(self.binary_model.classifier.children())[:-1]:
                embedding = layer(embedding)
            binary = self.binary_model.classifier[-1](embedding).reshape(-1)
            binary_classes = None
        outputs = {"binary": binary, "binary_classes": binary_classes}
        for name in selected:
            if name in SEMANTIC_TASKS:
                outputs[name] = getattr(self, name)(embedding)
            else:
                outputs[name] = deterministic_geometry_resize(
                    getattr(self, name)(spatial).sigmoid()
                )
        return outputs

    def forward(self, images):
        """Return the binary deployment logit without executing auxiliary heads."""
        return self.binary_model(images)


def multitask_loss(
    outputs, targets, *, auxiliary=True, weights=AuxiliaryLossWeights(), tasks=None
):
    """Combine binary and enabled auxiliary losses using the declared task weights.

    Return both the total loss and component values; missing/invalid targets fail.
    """
    if outputs["binary_classes"] is None:
        binary = F.binary_cross_entropy_with_logits(
            outputs["binary"].float(), targets["binary"].float()
        )
    else:
        binary = F.cross_entropy(outputs["binary_classes"].float(), targets["binary"])
    parts = {"binary": binary}
    if not auxiliary:
        return binary, parts
    selected = active_auxiliary_tasks(
        auxiliary=auxiliary,
        geometry=any(name in outputs for name in GEOMETRIC_TASKS),
        tasks=tasks,
    )
    for name in selected:
        if name == "face_attributes":
            parts[name] = F.binary_cross_entropy_with_logits(
                outputs[name].float(), targets[name].float()
            )
        elif name in ("spoof_type", "illumination"):
            parts[name] = F.cross_entropy(outputs[name].float(), targets[name])
    total = parts["binary"]
    if auxiliary:
        for name in SEMANTIC_TASKS:
            if name not in selected:
                continue
            total = total + getattr(weights, name) * parts[name]
    if any(name in selected for name in GEOMETRIC_TASKS):
        for name in GEOMETRIC_TASKS:
            if name not in selected:
                continue
            if name not in targets:
                raise ValueError(f"Missing verified geometric pseudo-label: {name}")
            target = targets[name].float()
            expected = outputs[name]
            if (
                target.shape != expected.shape
                or not torch.isfinite(target).all()
                or (target < 0).any()
                or (target > 1).any()
            ):
                raise ValueError(
                    f"Geometric target must match output shape and lie in [0,1]: {name}"
                )
            # The source method sets spoof depth and live reflection to zero.
            zero_rows = targets["binary"] == (1 if name == "depth_map" else 0)
            if torch.count_nonzero(target[zero_rows]):
                raise ValueError(f"Nonzero {name} for source zero-target class")
            parts[name] = F.mse_loss(expected.float(), target)
            total = total + getattr(weights, name) * parts[name]
    return total, parts

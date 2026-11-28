"""Native PyTorch inference for the cited CVPR-2018 reflection teacher.

Follows ceciliavision/perceptual-reflection-removal/main.py: VGG hypercolumns,
legacy TensorFlow bilinear coordinates, image-statistics BatchNorm and BGR I/O.
Weights come from the author's TensorFlow checkpoint and graph constants.
Numerical equivalence to an executed TensorFlow reference must be checked before
claiming an exact framework conversion.
"""

import torch
from torch import nn
from torch.nn import functional as F

VGG_BLOCK_LAYERS = (2, 2, 4, 4, 2)
GENERATOR_LAYERS = (
    "g_conv0",
    "g_conv1",
    "g_conv2",
    "g_conv3",
    "g_conv4",
    "g_conv5",
    "g_conv6",
    "g_conv7",
    "g_conv9",
)
GENERATOR_DILATIONS = (1, 1, 2, 4, 8, 16, 32, 64, 1)


def tensorflow_bilinear_resize(images, size):
    """TF1 ResizeBilinear: align_corners=False and half_pixel_centers=False."""
    result = images
    for axis, length in ((2, size[0]), (3, size[1])):
        coordinates = torch.arange(
            length, device=images.device, dtype=torch.float32
        ) * (result.shape[axis] / length)
        lower = coordinates.floor().long()
        upper = (lower + 1).clamp(max=result.shape[axis] - 1)
        shape = [1] * 4
        shape[axis] = length
        weight = (coordinates - lower).reshape(shape)
        result = (
            result.index_select(axis, lower) * (1 - weight)
            + result.index_select(axis, upper) * weight
        )
    return result


class ReflectionTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = nn.ModuleList()
        channels = 3
        for count, output in zip(
            VGG_BLOCK_LAYERS, (64, 128, 256, 512, 512), strict=True
        ):
            block = nn.ModuleList()
            for _ in range(count):
                block.append(nn.Conv2d(channels, output, 3, padding=1))
                channels = output
            self.vgg.append(block)
        self.generator = nn.ModuleList()
        for index, dilation in enumerate(GENERATOR_DILATIONS):
            kernel = 1 if index == 0 else 3
            self.generator.append(
                nn.Conv2d(
                    1475 if index == 0 else 64,
                    64,
                    kernel,
                    padding=0 if kernel == 1 else dilation,
                    dilation=dilation,
                    bias=False,
                )
            )
        self.beta = nn.ParameterList(
            [nn.Parameter(torch.zeros(64)) for _ in GENERATOR_LAYERS]
        )
        self.w0 = nn.ParameterList(
            [nn.Parameter(torch.ones(())) for _ in GENERATOR_LAYERS]
        )
        self.w1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in GENERATOR_LAYERS]
        )
        self.output = nn.Conv2d(64, 6, 1)
        self.register_buffer(
            "vgg_mean", torch.tensor([123.68, 116.779, 103.939]).reshape(1, 3, 1, 1)
        )

    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[1] != 3:
            raise ValueError(
                "Source reflection inference requires one NCHW RGB image at a time"
            )
        bgr = rgb[:, [2, 1, 0]]
        value = bgr * 255.0 - self.vgg_mean
        features = []
        for block_index, block in enumerate(self.vgg):
            for layer_index, layer in enumerate(block):
                value = layer(value).relu()
                if layer_index == 1:
                    features.append(value)
            if block_index < 4:
                value = F.avg_pool2d(
                    value,
                    kernel_size=2,
                    stride=2,
                    ceil_mode=True,
                    count_include_pad=False,
                )
        value = torch.cat(
            [
                tensorflow_bilinear_resize(feature, rgb.shape[2:]) / 255.0
                for feature in reversed(features)
            ]
            + [bgr],
            dim=1,
        )
        for index, layer in enumerate(self.generator):
            value = layer(value)
            mean = value.mean((0, 2, 3), keepdim=True)
            variance = (value - mean).square().mean((0, 2, 3), keepdim=True)
            # The source's nm() calls slim.batch_norm with its default
            # is_training=True even in testing; saved running stats are unused.
            normalized = (value - mean) * torch.rsqrt(variance + 0.001) + self.beta[
                index
            ].reshape(1, -1, 1, 1)
            value = self.w0[index] * value + self.w1[index] * normalized
            value = F.leaky_relu(value, negative_slope=0.2)
        reflection_bgr = self.output(value)[:, 3:]
        return reflection_bgr[:, [2, 1, 0]]

    @torch.inference_mode()
    def pseudo_reflection(self, rgb):
        # Match the author's saved uint8 reflection PNG before ToTensor.
        return (self(rgb).clamp(0, 1) * 255).floor() / 255.0

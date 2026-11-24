"""Small three-class CNN trained from scratch."""
import torch
from torch import nn


class EmotionCNN(nn.Module):
    def __init__(self, channels=(32, 64, 128), dropout=0.3):
        super().__init__()
        blocks = []
        previous = 1
        for width in channels:
            blocks.extend([nn.Conv2d(previous, width, 3, stride=1, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU(), nn.MaxPool2d(2, 2)])
            previous = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(previous, 3))

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 48, 48):
            raise ValueError('Expected float32 [B,1,48,48] input.')
        if images.dtype != torch.float32:
            raise ValueError('Expected float32 input.')
        return self.head(self.features(images))

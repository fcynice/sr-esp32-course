"""ESPCN-style super-resolution network.

Three convolutional layers followed by a sub-pixel convolution
(``nn.PixelShuffle``) upsampling layer.

Channel layout for RGB input and scale r:
    Conv1  5x5  3      -> 64    + ReLU
    Conv2  3x3  64     -> 32    + ReLU
    Conv3  3x3  32     -> 3*r^2 (no activation)
    PixelShuffle(r)    3*r^2 -> 3  (H*r, W*r)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ESPCN(nn.Module):
    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        mid1: int = 64,
        mid2: int = 32,
    ) -> None:
        super().__init__()
        if scale not in (2, 3, 4, 8):
            raise ValueError(f"Unsupported scale: {scale}")

        self.scale = scale
        self.in_channels = in_channels
        self.mid1 = mid1
        self.mid2 = mid2

        self.conv1 = nn.Conv2d(in_channels, mid1, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(mid1, mid2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(mid2, in_channels * scale**2, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        self._init_weights()

    def _init_weights(self) -> None:
        for conv in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        # Final layer predicts residual-ish sub-pixel values; keep it small.
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x


def build_model(**kwargs) -> ESPCN:
    return ESPCN(**kwargs)

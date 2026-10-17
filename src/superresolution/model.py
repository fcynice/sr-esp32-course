"""Super-resolution networks.

Models
------
``espcn``
    Three convolutional layers + ``nn.PixelShuffle`` upsampling.  Small and fast
    (~37K parameters); kept for backward compatibility with older checkpoints.

``edsr``
    Residual network in the spirit of EDSR (no BatchNorm).  Features are
    extracted in low-resolution space, then upsampled with one or more
    ``nn.PixelShuffle`` stages::

        head  Conv3x3 3 -> C
        body  N x ResidualBlock(C)          # x + conv2(relu(conv1(x)))
        fuse  conv_after(body) + head       # global residual (LR space)
        up    [Conv3x3 C -> C*4, PixelShuffle(2)] * log2(scale)
        tail  Conv3x3 C -> 3

    Default C=128, N=32 gives ~10.8M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# ESPCN
# ---------------------------------------------------------------------------


class ESPCN(nn.Module):
    """Three conv layers + PixelShuffle (ESPCN)."""

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
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="linear")
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x


# ---------------------------------------------------------------------------
# EDSR-style residual network
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Two convs + ReLU with an identity skip.

    The residual branch is scaled by ``res_scale`` (0.1) so that stacking many
    blocks does not make the activations grow exponentially at initialization.
    """

    def __init__(self, channels: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.conv2(self.act(self.conv1(x)))


class EDSR(nn.Module):
    """Lightweight EDSR: residual body + PixelShuffle upsampler."""

    def __init__(
        self,
        scale: int = 4,
        in_channels: int = 3,
        channels: int = 128,
        n_blocks: int = 32,
    ) -> None:
        super().__init__()
        if scale not in (2, 3, 4, 8):
            raise ValueError(f"Unsupported scale: {scale}")

        self.scale = scale
        self.in_channels = in_channels
        self.channels = channels
        self.n_blocks = n_blocks

        self.head = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_blocks)])
        self.conv_after_body = nn.Conv2d(channels, channels, 3, padding=1)

        ups: list[nn.Module] = []
        remaining = scale
        while remaining % 2 == 0:
            ups += [
                nn.Conv2d(channels, channels * 4, 3, padding=1),
                nn.PixelShuffle(2),
            ]
            remaining //= 2
        if remaining > 1:
            ups += [
                nn.Conv2d(channels, channels * remaining**2, 3, padding=1),
                nn.PixelShuffle(remaining),
            ]
        self.upsampler = nn.Sequential(*ups)
        self.tail = nn.Conv2d(channels, in_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Start with a small output so the initial L1 loss is low and stable.
        nn.init.normal_(self.tail.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        body = self.body(feat)
        body = self.conv_after_body(body) + feat
        out = self.upsampler(body)
        return self.tail(out)


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "espcn": ESPCN,
    "edsr": EDSR,
}


def build_model(name: str = "edsr", **kwargs) -> nn.Module:
    """Build a model by name with the given keyword arguments."""
    try:
        cls = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from None
    return cls(**kwargs)


def _get(obj, key, default):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def model_kwargs_for(model_name: str, args) -> dict:
    """Extract architecture kwargs from an argparse Namespace or a dict."""
    scale = _get(args, "scale", 4)
    if model_name == "espcn":
        return {
            "scale": scale,
            "in_channels": 3,
            "mid1": _get(args, "mid1", 64),
            "mid2": _get(args, "mid2", 32),
        }
    if model_name == "edsr":
        return {
            "scale": scale,
            "in_channels": 3,
            "channels": _get(args, "channels", 128),
            "n_blocks": _get(args, "n_blocks", 32),
        }
    return {"scale": scale}

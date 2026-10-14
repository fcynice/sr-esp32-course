"""Super-resolution package (ESPCN + PixelShuffle)."""

from .model import ESPCN

__all__ = ["ESPCN"]


def main() -> None:
    from .train import main as train_main

    train_main()

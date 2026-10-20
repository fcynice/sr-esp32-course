"""Versioned ESP32 camera degradation models.

Public API
----------
``degrade_image``
    Degrade a full HR PIL image and return ``(LR PIL image, spec dict)``.
``degrade_patch``
    Degrade an HR region and center-crop to a requested LR size.  Used by the
    on-the-fly training dataset.
``PROFILES`` / ``profile_names``
    Introspection used by the review CLI and visualiser.

Every degradation model is a separate module (``v1``, ``v2``, ...).  A version
that has been reviewed is frozen; new tuning means a new version module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from . import ops, v1

VERSIONS: dict[str, object] = {
    v1.VERSION: v1,
}

DEFAULT_VERSION = "v1"


def list_versions() -> tuple[str, ...]:
    return tuple(sorted(VERSIONS))


def get_version(version: str = DEFAULT_VERSION):
    try:
        return VERSIONS[version]
    except KeyError:
        raise KeyError(
            f"unknown degradation version: {version} (available: {', '.join(list_versions())})"
        ) from None


def fingerprint(version: str = DEFAULT_VERSION) -> str:
    """SHA-256 of the frozen version module and shared low-level ops.

    Stored in review manifests and training run metadata so a session can be
    tied to the exact code that produced it.  If this value changes for an old
    version, that version was modified and is no longer reproducible.
    """
    module = get_version(version)
    digest = hashlib.sha256()
    for raw in (getattr(module, "__file__", None), getattr(ops, "__file__", None)):
        if not raw:
            continue
        path = Path(raw)
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def profile_names(version: str = DEFAULT_VERSION) -> tuple[str, ...]:
    return tuple(get_version(version).profile_names())


def profile_weights(version: str = DEFAULT_VERSION) -> dict[str, float]:
    return dict(get_version(version).PROFILE_WEIGHTS)


def degrade_image(
    img: Image.Image,
    *,
    version: str = DEFAULT_VERSION,
    seed: int = 0,
    profile: str | None = None,
    scale: int = 4,
    severity: float = 1.0,
    output_lr_size: tuple[int, int] | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    """Degrade a full HR image.

    Parameters
    ----------
    img:
        HR RGB PIL image.
    version:
        Frozen degradation model version (``v1`` by default).
    seed:
        Seed for profile/parameter sampling.  Same seed + version + profile +
        severity + input image -> identical output.
    profile:
        Force a profile; when ``None`` a weighted random profile is chosen.
    scale:
        Integer downscale factor.  LR size is ``floor(HR / scale)`` unless
        ``output_lr_size`` is given.
    severity:
        Global strength multiplier (1.0 = reviewed default).
    output_lr_size:
        Optional centre-crop target in LR pixels.  Used by patch training.
    """
    module = get_version(version)
    rng = np.random.default_rng(int(seed))
    if profile is None:
        profile = module.choose_profile(rng)
    spec = module.sample_spec(profile, rng, severity)
    spec["scale"] = int(scale)
    spec["seed"] = int(seed)
    out = module.run(img, spec, rng, scale=scale)

    if output_lr_size is not None:
        tw, th = int(output_lr_size[0]), int(output_lr_size[1])
        if tw <= 0 or th <= 0:
            raise ValueError(f"invalid output_lr_size: {output_lr_size}")
        if out.width < tw or out.height < th:
            raise ValueError(
                f"degraded image {out.size} smaller than requested centre crop {(tw, th)}"
            )
        left = (out.width - tw) // 2
        top = (out.height - th) // 2
        out = out.crop((left, top, left + tw, top + th))
    return out, spec


def degrade_patch(
    hr_region: Image.Image,
    *,
    version: str = DEFAULT_VERSION,
    seed: int = 0,
    profile: str | None = None,
    scale: int = 4,
    severity: float = 1.0,
    output_lr_size: tuple[int, int] | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    """Alias with a clearer name for training-time patch degradation."""
    return degrade_image(
        hr_region,
        version=version,
        seed=seed,
        profile=profile,
        scale=scale,
        severity=severity,
        output_lr_size=output_lr_size,
    )


def choose_profiles_for_review(
    version: str = DEFAULT_VERSION,
    requested: Iterable[str] | None = None,
) -> list[str]:
    available = list(profile_names(version))
    if requested is None:
        return available
    wanted = [p for p in requested if p in available]
    if not wanted:
        raise ValueError(f"no valid profiles in {list(requested)!r}; available: {available}")
    return wanted

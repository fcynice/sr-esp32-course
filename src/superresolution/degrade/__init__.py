"""Versioned ESP32 camera degradation models.

Public API
----------
``degrade_image``
    Degrade a full HR PIL image and return ``(LR PIL image, spec dict)``.
``fingerprint``
    SHA-256 of the frozen version module and shared low-level ops.
``profile_names`` / ``profile_weights``
    Introspection used by the dataset generator.

Every degradation model is a separate module (``v1``, ``v2``, ...).  A version
that has been reviewed is frozen; new tuning means a new version module.

The package is intentionally batch-oriented: it writes pre-generated LR files
for the normal paired training pipeline.  It does not expose an on-the-fly
patch API.
"""


from __future__ import annotations

import hashlib
from pathlib import Path

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

    Stored in dataset manifests and training run metadata so files can be tied
    to the exact code that produced them.  If this value changes for an old
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
) -> tuple[Image.Image, dict[str, object]]:
    """Degrade a full HR image to ``floor(HR / scale)``.

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
        Integer downscale factor.
    severity:
        Global strength multiplier (1.0 = reviewed default).
    """
    module = get_version(version)
    rng = np.random.default_rng(int(seed))
    if profile is None:
        profile = module.choose_profile(rng)
    spec = module.sample_spec(profile, rng, severity)
    spec["scale"] = int(scale)
    spec["seed"] = int(seed)
    out = module.run(img, spec, rng, scale=scale)
    return out, spec

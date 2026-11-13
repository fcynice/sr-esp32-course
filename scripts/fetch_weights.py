#!/usr/bin/env python3
"""Download the released X4 checkpoints into ``runs/resNN/best.pth``.

Weights are published as GitHub Release assets (not committed to Git).  This
script downloads the two inference-only checkpoints, verifies their SHA-256 and
places them where ``superresolution.infer`` / the servers expect them.

Usage::

    .venv/bin/python scripts/fetch_weights.py
    .venv/bin/python scripts/fetch_weights.py --dest /tmp/runs --force

If the release has not been published yet, the GitHub URL returns 404.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = "fcynice/sr-esp32-course"
TAG = "weights-x4-v1"
BASE_URL = f"https://github.com/{REPO}/releases/download/{TAG}"

# asset name -> (target run directory, exact size, sha256)
ASSETS: dict[str, tuple[str, int, str]] = {
    "edsr_x4_bicubic.pth": (
        "res03",
        43171051,
        "2fa46dc547bccf5439ce072d1c653916e4d6af88d32ba769e1820e5c6fb0cc7b",
    ),
    "edsr_x4_esp32_v2.pth": (
        "res08",
        43171259,
        "a5724f75e5816845a5dbc9b592d5ee23f948e86dfd038a799426f98189e4deff",
    ),
}


@contextlib.contextmanager
def prefer_ipv4():
    """Prefer IPv4 for this request.

    Some CI/WSL networks have an IPv6 route that hangs instead of failing, while
    IPv4 works.  ``urllib`` otherwise picks the first address returned by the
    resolver, which can be IPv6.
    """

    original = socket.getaddrinfo

    def ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
        infos = original(host, port, socket.AF_UNSPEC, type, proto, flags)
        v4 = [info for info in infos if info[0] == socket.AF_INET]
        return v4 or infos

    socket.getaddrinfo = ipv4_first
    try:
        yield
    finally:
        socket.getaddrinfo = original


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_size: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "sr-esp32-course/1.0"})
    try:
        with prefer_ipv4(), urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or expected_size)
            done = 0
            with tmp.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    pct = 100.0 * done / max(1, total)
                    print(f"\r  {dest.name}: {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="")
            print()
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        if exc.code == 404:
            raise SystemExit(
                f"404: {url}\n"
                f"Release '{TAG}' 还没发布，或 asset 名称不对。"
            ) from None
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"download failed: {url}\n{exc}\n"
            f"Release '{TAG}' 可能还没发布，或网络无法访问 GitHub。"
        ) from None

    size = tmp.stat().st_size
    if size != expected_size:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"size mismatch for {dest.name}: {size} != {expected_size}")
    tmp.replace(dest)


def main() -> None:
    p = argparse.ArgumentParser(description="Download X4 checkpoints from GitHub Releases")
    p.add_argument("--dest", default="runs", help="runs directory (default: runs)")
    p.add_argument("--force", action="store_true", help="re-download even if the file exists")
    p.add_argument("--list", action="store_true", help="list assets and exit")
    args = p.parse_args()

    if args.list:
        for name, (run, size, digest) in ASSETS.items():
            print(f"{name}  ->  runs/{run}/best.pth  {size / 1e6:.1f} MB  sha256={digest}")
        return

    dest_root = Path(args.dest)
    for name, (run, size, expected_sha) in ASSETS.items():
        target = dest_root / run / "best.pth"
        if target.exists() and not args.force:
            actual = sha256(target)
            if actual == expected_sha:
                print(f"skip {target} (already downloaded)")
            else:
                print(f"skip {target} (exists, checksum differs; use --force to replace)")
            continue

        url = f"{BASE_URL}/{name}"
        print(f"download {url}")
        download(url, target, size)
        actual = sha256(target)
        if actual != expected_sha:
            raise SystemExit(f"sha256 mismatch for {target}: {actual}")
        print(f"ok {target}")


if __name__ == "__main__":
    main()

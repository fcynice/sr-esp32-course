#!/usr/bin/env python3
"""Robust, resumable, multi-connection file downloader.

Splits a remote file into N byte ranges, downloads them concurrently with
retries/backoff, then joins the parts. Falls back to a single stream when the
server does not support range requests.

Usage:
    python3 download_parallel.py URL OUTFILE [-c CONNECTIONS] [-r RETRIES]

Skips the download if OUTFILE already has the expected size (use --force to
re-download). Exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
import urllib.request

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CHUNK = 1 << 20  # 1 MiB read blocks


def _open(url: str, start: int | None = None, end: int | None = None, timeout: int = 120):
    headers = {"User-Agent": UA}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout)


def probe_size(url: str, retries: int = 8) -> tuple[int, bool]:
    """Return (size, supports_ranges)."""
    last = None
    for i in range(retries):
        try:
            with _open(url, 0, 0, timeout=60) as r:
                cr = r.headers.get("Content-Range")
                if r.status == 206 and cr:
                    return int(cr.split("/")[-1]), True
                cl = r.headers.get("Content-Length")
                if cl:
                    return int(cl), False
            last = "no Content-Length / Content-Range"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(min(2 ** i, 30))
    raise RuntimeError(f"probe failed: {last}")


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


class Progress:
    def __init__(self, total: int, label: str):
        self.total = max(total, 1)
        self.done = 0
        self.label = label
        self.lock = threading.Lock()
        self.start = time.time()
        self._last = 0.0

    def add(self, n: int) -> None:
        with self.lock:
            self.done += n
            now = time.time()
            if now - self._last < 0.5 and self.done < self.total:
                return
            self._last = now
            elapsed = max(now - self.start, 1e-6)
            pct = 100.0 * self.done / self.total
            rate = self.done / elapsed
            eta = (self.total - self.done) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  {self.label}: {pct:5.1f}%  "
                f"{_human(self.done)}/{_human(self.total)}  "
                f"{_human(rate)}/s  ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}   "
            )
            sys.stdout.flush()


def download_range(url: str, start: int, end: int, path: str,
                   retries: int, progress: Progress) -> None:
    expected = end - start + 1
    for attempt in range(retries + 1):
        got = 0
        try:
            with _open(url, start, end, timeout=300) as r, open(path, "wb") as fh:
                while True:
                    block = r.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    got += len(block)
                    progress.add(len(block))
            if got == expected:
                return
            raise IOError(f"short read {got} != {expected}")
        except Exception as e:  # noqa: BLE001
            progress.add(-got)  # roll back this attempt before retrying
            if attempt >= retries:
                raise RuntimeError(f"range {start}-{end} failed: {e}") from e
            time.sleep(min(2 ** attempt, 30))


def download(url: str, out: str, connections: int = 16, retries: int = 10,
             force: bool = False) -> None:
    label = os.path.basename(out)
    size, ranged = probe_size(url, retries=retries)
    if os.path.exists(out) and os.path.getsize(out) == size and not force:
        print(f"[skip] {out} already complete ({_human(size)})")
        return
    print(f"[get ] {label}  size={_human(size)}  ranges={ranged}  conns={connections}")
    tmp = out + ".parts"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    progress = Progress(size, label)
    t0 = time.time()
    try:
        if not ranged or connections <= 1:
            p = os.path.join(tmp, "0000")
            download_range(url, 0, size - 1, p, retries, progress)
            parts = [p]
        else:
            n = max(1, min(connections, size))
            step = (size + n - 1) // n
            jobs = []
            for i in range(n):
                s = i * step
                e = min(s + step - 1, size - 1)
                if s > e:
                    break
                jobs.append((s, e, os.path.join(tmp, f"{i:04d}")))
            errors: list[BaseException] = []

            def worker(s: int, e: int, p: str) -> None:
                try:
                    download_range(url, s, e, p, retries, progress)
                except BaseException as ex:  # noqa: BLE001
                    errors.append(ex)

            threads = [threading.Thread(target=worker, args=j, daemon=True) for j in jobs]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            if errors:
                raise errors[0]
            parts = [p for _, _, p in jobs]
        progress.add(0)  # force a final render
        sys.stdout.write("\n")
        with open(out, "wb") as dst:
            for p in parts:
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, dst, CHUNK)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    actual = os.path.getsize(out)
    if actual != size:
        os.remove(out)
        raise RuntimeError(f"size mismatch: expected {size}, got {actual}")
    elapsed = max(time.time() - t0, 1e-6)
    print(f"[ok  ] {out}  {_human(size)} in {elapsed:.1f}s ({_human(size / elapsed)}/s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("-c", "--connections", type=int, default=16)
    ap.add_argument("-r", "--retries", type=int, default=10)
    ap.add_argument("-f", "--force", action="store_true")
    args = ap.parse_args()
    download(args.url, args.out, args.connections, args.retries, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fast multi-connection downloader driven by `curl` subprocesses.

curl does the networking in C (much faster than Python threads for hundreds of
connections); this wrapper splits the file into N byte ranges, launches N curl
processes, shows live progress, retries short parts, and joins them in order.

Usage:
    python3 download_fast.py URL OUTFILE [-c CONNECTIONS] [-r RETRIES] [-f]

Skips the file when OUTFILE already has the expected size (unless --force).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

CURL = "curl"


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def head_size(url: str) -> int | None:
    try:
        out = subprocess.run(
            [CURL, "-sIL", "--max-time", "90", url],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception:
        return None
    for line in reversed(out.splitlines()):
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _curl_cmd(url: str, start: int, end: int, path: str, retries: int) -> list[str]:
    return [
        CURL, "-sSL",
        "--retry", str(retries), "--retry-delay", "2", "--retry-all-errors",
        "--connect-timeout", "30",
        "-r", f"{start}-{end}",
        "-o", path,
        url,
    ]


def download(url: str, out: str, connections: int = 256, retries: int = 10,
             force: bool = False) -> None:
    label = os.path.basename(out)
    size = head_size(url)
    if size is None:
        raise RuntimeError(f"could not determine size of {url}")
    if os.path.exists(out) and os.path.getsize(out) == size and not force:
        print(f"[skip] {out} already complete ({_human(size)})")
        return

    n = max(1, min(connections, size))
    step = (size + n - 1) // n
    tmp = out + ".parts"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    parts: list[tuple[int, int, str]] = []
    for i in range(n):
        s = i * step
        e = min(s + step - 1, size - 1)
        if s > e:
            break
        parts.append((s, e, os.path.join(tmp, f"{i:05d}")))

    print(f"[get ] {label}  size={_human(size)}  conns={len(parts)}")
    t0 = time.time()
    procs = [subprocess.Popen(_curl_cmd(url, s, e, p, retries))
             for s, e, p in parts]

    last = 0.0
    try:
        while True:
            alive = [pr for pr in procs if pr.poll() is None]
            done_bytes = 0
            for _, _, p in parts:
                try:
                    done_bytes += os.path.getsize(p)
                except OSError:
                    pass
            now = time.time()
            if now - last >= 0.5 or not alive:
                last = now
                elapsed = max(now - t0, 1e-6)
                pct = 100.0 * done_bytes / size
                rate = done_bytes / elapsed
                eta = (size - done_bytes) / rate if rate > 0 else 0
                sys.stdout.write(
                    f"\r  {label}: {pct:5.1f}%  {_human(done_bytes)}/{_human(size)}  "
                    f"{_human(rate)}/s  ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}   "
                )
                sys.stdout.flush()
            if not alive:
                break
            time.sleep(0.3)

        # verify each part and retry short ones once more
        for s, e, p in parts:
            want = e - s + 1
            got = os.path.getsize(p) if os.path.exists(p) else 0
            if got != want:
                print(f"\n  [retry] {os.path.basename(p)} {got}/{want}")
                subprocess.run(_curl_cmd(url, s, e, p, retries), check=False)
        sys.stdout.write("\n")

        bad = [(s, e, p) for s, e, p in parts
               if os.path.getsize(p) != e - s + 1]
        if bad:
            raise RuntimeError(f"{len(bad)} part(s) incomplete")

        with open(out, "wb") as dst:
            for _, _, p in parts:
                with open(p, "rb") as src:
                    shutil.copyfileobj(src, dst, 1 << 20)
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
    ap.add_argument("-c", "--connections", type=int, default=256)
    ap.add_argument("-r", "--retries", type=int, default=10)
    ap.add_argument("-f", "--force", action="store_true")
    args = ap.parse_args()
    download(args.url, args.out, args.connections, args.retries, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

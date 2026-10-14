#!/usr/bin/env bash
# Download all DIV2K *LR* archives (bicubic + unknown degradation, train+valid,
# X2/X3/X4) from the official ETH server into data/DIV2K.
#
# The official server throttles single connections (~0.26 MB/s) but scales with
# parallelism: 16->0.85, 64->2.83, 128->4.56, 256->9.7 MB/s. We therefore use
# many ranges per file (curl-driven). Re-running skips completed files.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/DIV2K"            # 解压后的数据集目录
ARCHIVE="$ROOT/data/archives/DIV2K" # 原始 zip 存放目录
DL="$ROOT/scripts/download_fast.py"
OFFICIAL="https://data.vision.ee.ethz.ch/cvl/DIV2K"
CONNS="${CONNS:-256}"
mkdir -p "$DEST" "$ARCHIVE"

FILES=(
  DIV2K_train_LR_bicubic_X2
  DIV2K_train_LR_bicubic_X3
  DIV2K_train_LR_bicubic_X4
  DIV2K_valid_LR_bicubic_X2
  DIV2K_valid_LR_bicubic_X3
  DIV2K_valid_LR_bicubic_X4
  DIV2K_train_LR_unknown_X2
  DIV2K_train_LR_unknown_X3
  DIV2K_train_LR_unknown_X4
  DIV2K_valid_LR_unknown_X2
  DIV2K_valid_LR_unknown_X3
  DIV2K_valid_LR_unknown_X4
)

log() { echo "[$(date '+%F %T')] $*"; }

log "=== DIV2K LR download start (conns=$CONNS) ==="
for f in "${FILES[@]}"; do
  ok=0
  for attempt in 1 2 3; do
    log "download $f (attempt $attempt)"
    if python3 "$DL" "$OFFICIAL/$f.zip" "$ARCHIVE/$f.zip" -c "$CONNS" -r 10; then
      ok=1; log "done $f"; break
    fi
    log "retry $f after failure"
    rm -f "$ARCHIVE/$f.zip"
    sleep 5
  done
  [ "$ok" = 1 ] || log "FAILED $f"
done
log "=== all LR downloads finished ==="
ls -lh "$ARCHIVE"/*.zip

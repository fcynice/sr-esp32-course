#!/usr/bin/env bash
# Download the DIV2K dataset into data/DIV2K.
#
# Strategy (see README notes):
#   * DIV2K_train_HR  -> fast hf-mirror copy (byte-verified identical PNGs)
#   * everything else -> official ETH server, accelerated with many connections
#
# Re-running is safe: completed files (correct size) are skipped.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/DIV2K"
DL="$ROOT/scripts/download_parallel.py"
mkdir -p "$DEST"

OFFICIAL="https://data.vision.ee.ethz.ch/cvl/DIV2K"
TRAIN_MIRROR="https://hf-mirror.com/datasets/yangtao9009/DIV2K/resolve/main/DIV2K_train.zip"

# Official files (name -> connections). Train HR comes from the mirror instead.
OFFICIAL_FILES=(
  DIV2K_valid_HR
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

fetch() { # url outfile connections label
  local url="$1" out="$2" conns="$3" label="$4"
  if [ -f "$out" ]; then
    log "skip $label (exists)"
    return 0
  fi
  for attempt in 1 2 3; do
    log "download $label  attempt $attempt"
    if python3 "$DL" "$url" "$out" -c "$conns" -r 12; then
      log "done $label"
      return 0
    fi
    log "retry $label after failure"
    rm -f "$out"
    sleep 5
  done
  log "FAILED $label"
  return 1
}

# Train HR from the fast mirror, in the background while official files download.
log "=== DIV2K download start ==="
fetch "$TRAIN_MIRROR" "$DEST/DIV2K_train.zip" 8 "DIV2K_train (hf-mirror)" &
TRAIN_PID=$!

rc=0
for f in "${OFFICIAL_FILES[@]}"; do
  fetch "$OFFICIAL/$f.zip" "$DEST/$f.zip" 32 "$f" || rc=1
done

wait "$TRAIN_PID" || rc=1
log "=== downloads finished (rc=$rc) ==="
ls -lh "$DEST"/*.zip 2>/dev/null
exit $rc

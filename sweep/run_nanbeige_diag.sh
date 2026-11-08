#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/oss-save/wangjunqi/neuron-explorer/looped-smoke/nanbeige-diag}
NAS_MODEL=${NANBEIGE_NAS:-/mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop-nanbeige/models/Nanbeige4.2-3B}
LOCAL_MODEL=${LOCAL_MODEL:-/tmp/wangjunqi-nanbeige-diag}
PYTHON=${PYTHON:-python}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"

mkdir -p "$RUN_DIR"
rm -rf -- "$LOCAL_MODEL"
mkdir -p "$LOCAL_MODEL"
cp -a "$NAS_MODEL"/. "$LOCAL_MODEL"/
cleanup() {
  rm -rf -- "$LOCAL_MODEL"
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
"$PYTHON" -m sweep.nanbeige_diag --model "$LOCAL_MODEL" --out "$RUN_DIR" \
  --batch-size "${BATCH_SIZE:-256}" --max-tokens "${MAX_TOKENS:-8}" --seed "${SEED:-0}"
printf 'Artifacts: %s\n' "$RUN_DIR"

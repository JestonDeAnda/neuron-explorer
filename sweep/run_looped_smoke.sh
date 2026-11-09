#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/oss-save/wangjunqi/looped-smoke}
LOCAL_ROOT=${LOCAL_ROOT:-/tmp/wangjunqi-looped-smoke-models}
PYTHON=${PYTHON:-python}
MAX_TOKENS=${MAX_TOKENS:-8}
SEED=${SEED:-0}
HUGINN_RECURRENCES=${HUGINN_RECURRENCES:-32}
FAMILIES=${FAMILIES:-ouro,nanbeige,huginn}

OURO_NAS=${OURO_NAS:-/mnt/nas-save/nas-save/wangjunqi/ouro/models/Ouro-2.6B}
NANBEIGE_NAS=${NANBEIGE_NAS:-/mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop-nanbeige/models/Nanbeige4.2-3B}
HUGINN_NAS=${HUGINN_NAS:-/mnt/nas-save/nas-save/wangjunqi/j-lens-on-loop/models/huginn-0125}

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR" "$LOCAL_ROOT"

stage_model() {
  local source="$1" destination="$2"
  rm -rf -- "$destination"
  (cd "$ROOT_DIR" && "$PYTHON" -c 'from sweep.looped_smoke import stage_directory; import sys; stage_directory(sys.argv[1], sys.argv[2])' "$source" "$destination")
}

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  for model_path in "$OURO_NAS" "$NANBEIGE_NAS" "$HUGINN_NAS"; do
    test -f "$model_path/config.json"
    find "$model_path" -maxdepth 1 -name '*.safetensors' -size +1M | grep -q .
    printf 'OK model %s\n' "$model_path"
  done
  probe="$RUN_DIR/mount-check.txt"
  printf 'ok %s\n' "$(date -Is)" >"$probe"
  rm -f "$probe"
  printf 'OK output %s\n' "$RUN_DIR"
  exit 0
fi

monitor_pid=""
local_model=""
stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
cleanup_local() {
  if [[ -n "$local_model" && -d "$local_model" ]]; then
    rm -rf -- "$local_model"
  fi
  local_model=""
}
trap 'stop_monitor; cleanup_local' EXIT INT TERM

specs=(
  "ouro|$OURO_NAS|256|"
  "nanbeige|$NANBEIGE_NAS|256|"
  "huginn|$HUGINN_NAS|64|$HUGINN_RECURRENCES"
)
failed=0
for spec in "${specs[@]}"; do
  IFS='|' read -r family nas_model batch recurrences <<< "$spec"
  case ",${FAMILIES}," in
    *",${family},"*) ;;
    *) continue ;;
  esac
  model_dir="$RUN_DIR/$family"
  mkdir -p "$model_dir"
  local_model="$LOCAL_ROOT/$family"
  cleanup_local
  local_model="$LOCAL_ROOT/$family"

  if ! stage_model "$nas_model" "$local_model" >"$model_dir/stage.log" 2>&1; then
    printf '%s\n' "staging failed: $nas_model" >>"$model_dir/smoke.log"
    printf '{"schema_version":1,"status":"failed","family":"%s","error":"staging failed"}\n' \
      "$family" >"$model_dir/smoke.json"
    failed=1
    cleanup_local
    continue
  fi

  metrics="$model_dir/nvidia.csv"
  (
    while true; do
      nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits >>"$metrics" 2>>"$model_dir/nvidia.log" || true
      sleep 1
    done
  ) &
  monitor_pid=$!

  command=("$PYTHON" -m sweep.looped_smoke
    --model "$local_model" --family "$family" --out "$model_dir"
    --batch-size "$batch" --max-tokens "$MAX_TOKENS" --seed "$SEED")
  if [[ -n "$recurrences" ]]; then
    command+=(--recurrences "$recurrences")
  fi
  (cd "$ROOT_DIR" && "${command[@]}") >"$model_dir/smoke.log" 2>&1
  rc=$?
  stop_monitor
  if (( rc != 0 )); then
    failed=1
  fi
  cleanup_local
 done

RUN_DIR="$RUN_DIR" FAMILIES="$FAMILIES" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from sweep.looped_smoke import aggregate_nvidia_csv, build_summary

root = Path(os.environ["RUN_DIR"])
records = []
selected = [item for item in os.environ.get("FAMILIES", "ouro,nanbeige,huginn").split(",") if item]
batches = {"ouro": 256, "nanbeige": 256, "huginn": 64}
for family in selected:
    requested_batch = batches[family]
    directory = root / family
    smoke_path = directory / "smoke.json"
    try:
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    except Exception as exc:
        smoke = {"status": "failed", "error": f"missing/invalid smoke.json: {exc}"}
    metrics_path = directory / "nvidia.csv"
    metrics = aggregate_nvidia_csv(metrics_path)
    probe = smoke.get("occupancy_probe", {})
    records.append({
        "family": family,
        "status": smoke.get("status", "failed"),
        "requested_batch": requested_batch,
        "actual_batch": probe.get("actual_batch"),
        "throughput_tokens_per_second": probe.get("throughput_tokens_per_second"),
        "cuda_peak_allocated_bytes": probe.get("cuda_peak_allocated_bytes"),
        "cuda_peak_reserved_bytes": probe.get("cuda_peak_reserved_bytes"),
        "nvidia": metrics,
        "smoke_json": str(smoke_path),
        "log": str(directory / "smoke.log"),
        "error": smoke.get("error"),
    })
(root / "summary.json").write_text(
    json.dumps(build_summary(records), indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
summary_rc=$?
if (( summary_rc != 0 )); then
  failed=1
fi

printf 'Artifacts: %s\n' "$RUN_DIR"
exit "$failed"

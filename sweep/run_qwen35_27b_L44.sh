#!/usr/bin/env bash
# Resume-safe Qwen3.5-27B layer-44 sweep. Does not submit a DLC job.
#
# MODEL must be a complete local safetensors snapshot (the DLC command stages one
# on the worker). The DSW copy /mnt/oss-all/models/Qwen3.5-27B is incomplete and
# is refused. Prompts and doses are steer_sweep.py's main_rel design with
# --dose-unit rel. Capture is off. The full pass is every neuron plus 256 random
# controls; a one-neuron smoke runs first and is skipped once marked successful.
#
#   MODEL=/path/to/complete/snapshot sweep/run_qwen35_27b_L44.sh
#   DRY_RUN=1 MODEL=/path/to/complete/snapshot sweep/run_qwen35_27b_L44.sh
#   sweep/run_qwen35_27b_L44.sh --verify-snapshot /path/to/snapshot
set -euo pipefail

DSW_SNAPSHOT="/mnt/oss-all/models/Qwen3.5-27B"
LAYER=44
TOKENS=2000000
SMOKE_NEURON=0
FULL_CONTROLS=256

verify_snapshot() {
  python3 - "$1" <<'PY'
import json
import os
import re
import struct
import sys

root = os.path.abspath(sys.argv[1])
SHARD_RE = re.compile(r"^(?P<prefix>.*)-(?P<idx>\d+)-of-(?P<total>\d+)\.safetensors$")


def fail(msg):
    print(f"INCOMPLETE: {root}: {msg}")
    sys.exit(1)


if not os.path.isdir(root):
    fail("not a directory")


def load_json(name):
    path = os.path.join(root, name)
    if not os.path.isfile(path):
        fail(f"missing {name}")
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        fail(f"invalid {name} ({exc})")


cfg = load_json("config.json")
if cfg.get("model_type") != "qwen3_5":
    fail(f"model_type is {cfg.get('model_type')!r}, expected 'qwen3_5'")
text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
try:
    n_layers = int(text["num_hidden_layers"])
except (KeyError, TypeError, ValueError):
    fail("config has no text num_hidden_layers")
if n_layers < 45:
    fail(f"num_hidden_layers={n_layers}; layer 44 is absent")
try:
    intermediate = int(text["intermediate_size"])
except (KeyError, TypeError, ValueError):
    fail("config has no text intermediate_size")
if intermediate < 1:
    fail(f"intermediate_size={intermediate}")

for name in ("tokenizer_config.json", "tokenizer.json"):
    path = os.path.join(root, name)
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        fail(f"missing or empty {name}")

for name in os.listdir(root):
    if name.endswith((".incomplete", ".aria2")):
        fail(f"partial download marker {name}")


def check_shard(path):
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return str(exc)
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            return f"{os.path.basename(path)}: truncated header"
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 1 or header_len > 50_000_000 or size < 8 + header_len:
            return f"{os.path.basename(path)}: header length {header_len} does not fit in {size} bytes"
        header = handle.read(header_len)
    try:
        meta = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        return f"{os.path.basename(path)}: safetensors header is not JSON ({exc})"
    if not isinstance(meta, dict):
        return f"{os.path.basename(path)}: safetensors header is not an object"
    end = 0
    tensors = 0
    for key, value in meta.items():
        if key == "__metadata__":
            continue
        tensors += 1
        if not isinstance(value, dict) or "data_offsets" not in value:
            return f"{os.path.basename(path)}: tensor {key} missing data_offsets"
        offsets = value["data_offsets"]
        try:
            start, stop = int(offsets[0]), int(offsets[1])
        except (TypeError, ValueError, IndexError):
            return f"{os.path.basename(path)}: tensor {key} has bad data_offsets"
        if start < 0 or stop < start:
            return f"{os.path.basename(path)}: tensor {key} offsets {start}:{stop} are invalid"
        end = max(end, stop)
    if tensors == 0:
        return f"{os.path.basename(path)}: no tensors in header"
    expected = 8 + header_len + end
    if size != expected:
        return (
            f"{os.path.basename(path)}: size {size} != header span {expected} "
            "(truncated or padded file)"
        )
    return None


def disk_groups():
    groups = {}
    for entry in os.listdir(root):
        match = SHARD_RE.match(entry)
        if not match:
            continue
        key = (match.group("prefix"), int(match.group("total")))
        groups.setdefault(key, set()).add(int(match.group("idx")))
    return groups


index_path = os.path.join(root, "model.safetensors.index.json")
single_path = os.path.join(root, "model.safetensors")
on_disk = disk_groups()

if os.path.isfile(index_path):
    try:
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, ValueError) as exc:
        fail(f"invalid model.safetensors.index.json ({exc})")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        fail("model.safetensors.index.json has an empty weight_map")
    shard_names = []
    seen = set()
    indexed = {}
    for tensor, shard in weight_map.items():
        if not isinstance(shard, str) or os.path.basename(shard) != shard or shard in ("", ".", ".."):
            fail(f"weight_map[{tensor!r}] is not a plain shard filename: {shard!r}")
        if shard not in seen:
            seen.add(shard)
            shard_names.append(shard)
        match = SHARD_RE.match(shard)
        if match:
            key = (match.group("prefix"), int(match.group("total")))
            indexed.setdefault(key, set()).add(int(match.group("idx")))
    for shard in shard_names:
        path = os.path.join(root, shard)
        if not os.path.isfile(path):
            fail(f"index lists missing shard {shard}")
        problem = check_shard(path)
        if problem:
            fail(problem)
    for (prefix, total), have in indexed.items():
        need = set(range(1, total + 1))
        if have != need:
            fail(f"index {prefix}-of-{total:05d} is missing shards {sorted(need - have)}")
    for (prefix, total), have in on_disk.items():
        need = set(range(1, total + 1))
        if have != need:
            fail(f"on-disk {prefix}-of-{total:05d} has {sorted(have)}, missing {sorted(need - have)}")
        if indexed.get((prefix, total), set()) != need:
            fail(f"index does not list every shard of {prefix}-of-{total:05d}")
    print(
        f"COMPLETE: {root}: {len(shard_names)} safetensors shards, "
        f"{len(weight_map)} tensors, layers={n_layers}, intermediate={intermediate}"
    )
    sys.exit(0)

if on_disk:
    listed = sorted(entry for entry in os.listdir(root) if SHARD_RE.match(entry))
    fail("sharded safetensors are present but model.safetensors.index.json is missing: " + ", ".join(listed))
if os.path.isfile(single_path):
    problem = check_shard(single_path)
    if problem:
        fail(problem)
    print(f"COMPLETE: {root}: single model.safetensors, layers={n_layers}, intermediate={intermediate}")
    sys.exit(0)
fail("no model.safetensors.index.json and no model.safetensors")
PY
}

refuse_dsw_model() {
  local model="${1%/}"
  local dsw="${DSW_SNAPSHOT%/}"
  if [[ "$model" == "$dsw" ]]; then
    return 0
  fi
  if [[ -e "$model" && -e "$dsw" ]] && [[ "$(readlink -f -- "$model")" == "$(readlink -f -- "$dsw")" ]]; then
    return 0
  fi
  return 1
}

check_corpus() {
  python3 - "$1" <<'PY'
import sys

path = sys.argv[1]
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print(f"INCOMPLETE: {path}: pyarrow is required to validate the corpus schema", file=sys.stderr)
    sys.exit(1)


def is_string(dtype):
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or pa.types.is_string_view(dtype):
        return True
    if pa.types.is_dictionary(dtype):
        return is_string(dtype.value_type)
    return False


try:
    schema = pq.read_schema(path)
except Exception as exc:
    print(f"INCOMPLETE: {path}: unreadable parquet ({exc})", file=sys.stderr)
    sys.exit(1)
if "text" not in schema.names:
    found = ", ".join(schema.names) or "(none)"
    print(f"INCOMPLETE: {path}: missing text column; columns: {found}", file=sys.stderr)
    sys.exit(1)
dtype = schema.field("text").type
if not is_string(dtype):
    print(f"INCOMPLETE: {path}: text column must be string, got {dtype}", file=sys.stderr)
    sys.exit(1)
print(f"CORPUS_OK: {path}: text={dtype}")
PY
}

if [[ "${1:-}" == "--verify-snapshot" ]]; then
  if [[ $# -ne 2 ]]; then
    echo "usage: $0 --verify-snapshot MODEL_DIR" >&2
    exit 2
  fi
  verify_snapshot "$2"
  exit 0
fi

if [[ $# -ne 0 ]]; then
  echo "usage: MODEL=/path/to/complete/snapshot $0" >&2
  echo "       DRY_RUN=1 MODEL=/path/to/complete/snapshot $0" >&2
  echo "       $0 --verify-snapshot MODEL_DIR" >&2
  exit 2
fi

MODEL="${MODEL:-}"
CORPUS="${CORPUS:-/mnt/oss-dataset/Nemotron_CC_v2.1/High-Quality/part_000000.parquet}"
OUT="${OUT:-/mnt/oss-save/wangjunqi/neuron-explorer/qwen35-27b-L44}"
CODE_ROOT="${CODE_ROOT:-/mnt/nas-save/wangjunqi/neuron-explorer-qwen35-27b}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$MODEL" ]]; then
  echo "MODEL is required. Point it at a node-local complete Qwen3.5-27B snapshot." >&2
  echo "The DSW snapshot $DSW_SNAPSHOT is incomplete and is not a default." >&2
  exit 2
fi

if refuse_dsw_model "$MODEL"; then
  echo "NOT USABLE: $MODEL is the known incomplete DSW snapshot $DSW_SNAPSHOT." >&2
  echo "It has no complete safetensors index. A partial first shard is not a loadable model." >&2
  echo "Stage a complete snapshot on the worker and pass that path as MODEL." >&2
  verify_snapshot "$MODEL" || true
  exit 2
fi

verify_snapshot "$MODEL"
check_corpus "$CORPUS"

for rel in sweep/neuron_stats.py sweep/steer_sweep.py sweep/rank.py; do
  if [[ ! -f "$CODE_ROOT/$rel" ]]; then
    echo "code checkout is missing $CODE_ROOT/$rel" >&2
    exit 2
  fi
done

if [[ -x "$CODE_ROOT/.venv/bin/python" ]]; then
  PY="$CODE_ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  PY="$(command -v python3)"
fi

STATS="$OUT/stats/qwen35-27b_L${LAYER}.npz"
SMOKE_OK="$OUT/runs/qwen35-27b-L44-smoke.ok"
SMOKE_JSONL="$OUT/runs/qwen35-27b-L44-smoke.jsonl"
SMOKE_META="$OUT/runs/qwen35-27b-L44-smoke.meta.json"
FULL_JSONL="$OUT/runs/qwen35-27b-L44.jsonl"
FULL_META="$OUT/runs/qwen35-27b-L44.meta.json"
UNITS="$OUT/runs/qwen35-27b-L44.units.jsonl"
PIPELINE_LOG="$OUT/logs/pipeline.log"

stats_cmd=(
  "$PY" sweep/neuron_stats.py
  --model "$MODEL"
  --layers "$LAYER"
  --tokens "$TOKENS"
  --corpus-parquet "$CORPUS"
  --out "$OUT/stats/qwen35-27b"
)
smoke_cmd=(
  "$PY" sweep/steer_sweep.py
  --model "$MODEL"
  --stats "$STATS"
  --design main_rel
  --dose-unit rel
  --neurons "$SMOKE_NEURON"
  --random-controls 0
  --capture-layers ""
  --capture-root "$OUT/captures"
  --max-num-seqs 320
  --run qwen35-27b-L44-smoke
  --out-dir "$OUT/runs"
)
full_cmd=(
  "$PY" sweep/steer_sweep.py
  --model "$MODEL"
  --stats "$STATS"
  --design main_rel
  --dose-unit rel
  --neurons all
  --random-controls "$FULL_CONTROLS"
  --capture-layers ""
  --capture-root "$OUT/captures"
  --max-num-seqs 320
  --run qwen35-27b-L44
  --out-dir "$OUT/runs"
)
rank_cmd=(
  "$PY" sweep/rank.py
  "$FULL_JSONL"
  --out "$UNITS"
  --show 40
)

quote_cmd() {
  printf 'CMD:'
  printf ' %q' "$@"
  printf '\n'
}

smoke_is_current=0
if [[ -f "$SMOKE_OK" ]] && grep -qxF "model=$MODEL" "$SMOKE_OK"; then
  smoke_is_current=1
fi

echo "host=$(hostname) date=$(date -Is) dry_run=$DRY_RUN"
echo "MODEL=$MODEL"
echo "CORPUS=$CORPUS"
echo "OUT=$OUT"
echo "CODE_ROOT=$CODE_ROOT"
echo "python=$PY"
if [[ "$smoke_is_current" -eq 1 ]]; then
  echo "smoke: skip (already marked successful for this MODEL at $SMOKE_OK)"
else
  echo "smoke: run neuron $SMOKE_NEURON once, then mark $SMOKE_OK"
fi
echo "full: layer $LAYER neurons=all random-controls=$FULL_CONTROLS design=main_rel dose-unit=rel capture=disabled tokens=$TOKENS"
quote_cmd "${stats_cmd[@]}"
quote_cmd "${smoke_cmd[@]}"
quote_cmd "${full_cmd[@]}"
quote_cmd "${rank_cmd[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN: snapshot and corpus checked; pipeline was not started."
  exit 0
fi

case "$OUT" in
  /mnt/oss-save/*) ;;
  *)
    echo "Refusing OUT=$OUT. Sweep outputs must stay under /mnt/oss-save." >&2
    exit 2
    ;;
esac

mkdir -p "$OUT/stats" "$OUT/runs" "$OUT/logs" "$OUT/captures"
export PYTHONUNBUFFERED=1

run_logged() {
  local log="$1"
  shift
  quote_cmd "$@"
  "$@" 2>&1 | tee -a "$log" "$PIPELINE_LOG"
}

cd "$CODE_ROOT"

meta_model() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("model",""))' "$1"
}

if [[ -f "$STATS" ]]; then
  python3 - "$MODEL" "$STATS" <<'PY'
import json
import os
import sys

import numpy as np

model, stats_path = sys.argv[1], sys.argv[2]
with open(os.path.join(model, "config.json"), encoding="utf-8") as handle:
    cfg = json.load(handle)
text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
intermediate = int(text["intermediate_size"])
with np.load(stats_path) as blob:
    layer = int(blob["layer"])
    neurons = int(blob["abs_max"].shape[0])
    tokens = int(blob["tokens"])
    saved = str(blob["model"])
if layer != 44:
    sys.exit(f"refusing {stats_path}: layer {layer} != 44")
if neurons != intermediate:
    sys.exit(f"refusing {stats_path}: abs_max {neurons} != config intermediate_size {intermediate}")
if tokens < 2000000:
    sys.exit(f"refusing {stats_path}: tokens {tokens} < 2000000")
if os.path.abspath(saved) != os.path.abspath(model) and saved != model:
    sys.exit(f"refusing {stats_path}: stats model {saved!r} != MODEL {model!r}")
print(f"stats reuse ok: {stats_path} neurons={neurons} tokens={tokens}")
PY
  echo "stats: reuse $STATS" | tee -a "$PIPELINE_LOG"
else
  run_logged "$OUT/logs/stats_L44.log" "${stats_cmd[@]}"
fi

if [[ -f "$SMOKE_META" ]]; then
  smoke_recorded="$(meta_model "$SMOKE_META")"
  if [[ "$smoke_recorded" != "$MODEL" ]]; then
    echo "smoke outputs belong to $smoke_recorded; removing them before a new smoke" | tee -a "$PIPELINE_LOG"
    rm -f "$SMOKE_JSONL" "$SMOKE_META" "$SMOKE_OK"
    smoke_is_current=0
  fi
fi

if [[ "$smoke_is_current" -eq 1 ]]; then
  echo "smoke already marked successful for this MODEL; not rerunning" | tee -a "$PIPELINE_LOG"
else
  run_logged "$OUT/logs/sweep_smoke.log" "${smoke_cmd[@]}"
  {
    echo "ok $(date -Is)"
    echo "model=$MODEL"
    echo "neuron=$SMOKE_NEURON"
    echo "design=main_rel"
    echo "dose_unit=rel"
    echo "random_controls=0"
    echo "capture=disabled"
  } > "$SMOKE_OK"
  sync -f "$SMOKE_OK" 2>/dev/null || true
  echo "smoke marked successful: $SMOKE_OK" | tee -a "$PIPELINE_LOG"
fi

if [[ -f "$FULL_META" ]]; then
  full_recorded="$(meta_model "$FULL_META")"
  if [[ "$full_recorded" != "$MODEL" ]]; then
    echo "full run meta model ($full_recorded) does not match MODEL ($MODEL); refusing to resume" >&2
    exit 2
  fi
fi

run_logged "$OUT/logs/sweep_L44.log" "${full_cmd[@]}"

{
  echo "model=$MODEL"
  echo "corpus=$CORPUS"
  echo "stats=$STATS"
  echo "jsonl=$FULL_JSONL"
  echo "layer=$LAYER"
  echo "design=main_rel"
  echo "dose_unit=rel"
  echo "neurons=all"
  echo "random_controls=$FULL_CONTROLS"
  echo "capture=disabled"
  echo "tokens=$TOKENS"
  "${rank_cmd[@]}"
} 2>&1 | tee -a "$OUT/logs/rank.log" "$PIPELINE_LOG" | tee "$OUT/summary.txt"

echo "PIPELINE_DONE $(date -Is) summary=$OUT/summary.txt units=$UNITS" | tee -a "$PIPELINE_LOG"

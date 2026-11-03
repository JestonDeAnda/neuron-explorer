#!/usr/bin/env bash
# CPU DLC mount check for the Qwen3.5-27B L44 sweep. Does not submit a job.
# A missing or partial model is reported and does not fail the check: the GPU
# job may download a complete snapshot onto node-local storage.
#
#   sweep/check_qwen35_27b_mounts.sh
#   CODE_ROOT=... DATA=... OUT=... sweep/check_qwen35_27b_mounts.sh
set -u

CODE_ROOT="${CODE_ROOT:-/mnt/nas-save/wangjunqi/neuron-explorer-qwen35-27b}"
DATA="${DATA:-/mnt/oss-dataset/Nemotron_CC_v2.1/High-Quality/part_000000.parquet}"
OUT="${OUT:-/mnt/oss-save/wangjunqi/neuron-explorer/qwen35-27b-L44}"

fail=0
ok() { printf 'OK: %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

printf 'CODE_ROOT=%s\nDATA=%s\nOUT=%s\n' "$CODE_ROOT" "$DATA" "$OUT"

for rel in sweep/neuron_stats.py sweep/steer_sweep.py sweep/rank.py sweep/run_qwen35_27b_L44.sh; do
  path="$CODE_ROOT/$rel"
  if [[ -f "$path" && -r "$path" && -s "$path" ]]; then
    ok "code $path"
  else
    bad "missing code $path"
  fi
done

if [[ ! -f "$DATA" ]]; then
  bad "parquet missing: $DATA"
elif [[ ! -r "$DATA" || ! -s "$DATA" ]]; then
  bad "parquet not readable: $DATA"
else
  ok "parquet exists and is readable ($(stat -c%s "$DATA" 2>/dev/null || echo '?') bytes): $DATA"
fi

probe="$OUT/.qwen35_27b_mountcheck.$$"
if mkdir -p -- "$OUT" \
  && printf 'mountcheck\n' > "$probe" \
  && grep -qx 'mountcheck' -- "$probe" \
  && rm -f -- "$probe"; then
  ok "OUT mkdir/write/delete: $OUT"
else
  bad "OUT mkdir/write/delete failed: $OUT"
  rm -f -- "$probe" 2>/dev/null || true
fi

if ! command -v python3 >/dev/null 2>&1; then
  bad "python3 is not available; cannot check the parquet text column or model shards"
  exit "$fail"
fi

if [[ -f "$DATA" && -r "$DATA" && -s "$DATA" ]]; then
  python3 - "$DATA" <<'PY'
import sys

path = sys.argv[1]
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print(f"WARN: pyarrow unavailable; did not check text column: {path}")
    sys.exit(0)

is_string_view = getattr(pa.types, "is_string_view", lambda _dtype: False)


def is_string(dtype):
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype) or is_string_view(dtype):
        return True
    if pa.types.is_dictionary(dtype):
        return is_string(dtype.value_type)
    return False


try:
    schema = pq.read_schema(path)
except Exception as exc:
    print(f"FAIL: parquet unreadable: {path}: {exc}")
    sys.exit(1)
if "text" not in schema.names:
    found = ", ".join(schema.names) or "(none)"
    print(f"FAIL: parquet missing text column: {path}: columns: {found}")
    sys.exit(1)
dtype = schema.field("text").type
if not is_string(dtype):
    print(f"FAIL: parquet text column is {dtype}, expected string: {path}")
    sys.exit(1)
print(f"OK: parquet text column {dtype}: {path}")
PY
  py_rc=$?
  if [[ "$py_rc" -ne 0 ]]; then
    fail=1
  fi
fi

python3 - <<'PY'
import json
import os

MIN_SHARD = 1 << 20
MIN_INDEX = 128
KNOWN = "/mnt/oss-all/models/Qwen3.5-27B"
NAMES = ("Qwen3.5-27B", "models--Qwen--Qwen3.5-27B")
PARENTS = (
    "",
    "models",
    "models/Qwen",
    "modelscope/Qwen",
    "modelscope/models/Qwen",
    "huggingface/Qwen",
    "huggingface/hub",
    "models/huggingface/hub",
    "hub",
)

print(f"shard nontrivial threshold: {MIN_SHARD} bytes")


def mounts():
    found = []
    if os.path.isdir("/mnt"):
        for name in sorted(os.listdir("/mnt")):
            path = os.path.join("/mnt", name)
            if os.path.isdir(path):
                found.append(path)
    return found


def is_known_dsw(root):
    try:
        real_root = os.path.realpath(root).rstrip("/")
        real_known = os.path.realpath(KNOWN).rstrip("/")
    except OSError:
        real_root = os.path.abspath(root).rstrip("/")
        real_known = os.path.abspath(KNOWN).rstrip("/")
    return real_root == real_known or real_root.startswith(real_known + "/")


def expand(path):
    snap = os.path.join(path, "snapshots")
    if os.path.isdir(snap):
        kids = []
        try:
            names = sorted(os.listdir(snap))
        except OSError as exc:
            print(f"WARN: cannot list {snap}: {exc}")
            return [path]
        for name in names:
            kid = os.path.join(snap, name)
            if os.path.isdir(kid):
                kids.append(kid)
        if kids:
            return kids
    return [path]


candidates = []
seen = set()
for mount in mounts():
    for parent in PARENTS:
        base = mount if parent == "" else os.path.join(mount, parent)
        if parent and not os.path.isdir(base):
            continue
        for name in NAMES:
            path = os.path.join(base, name)
            if not os.path.isdir(path):
                continue
            for cand in expand(path):
                try:
                    key = os.path.realpath(cand)
                except OSError:
                    key = os.path.abspath(cand)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(cand)


def file_size(path):
    try:
        return os.path.getsize(path), None
    except OSError as exc:
        return None, str(exc)


def report(root):
    print(f"MODEL: {root}")
    if is_known_dsw(root):
        print(
            "  note: known incomplete DSW snapshot "
            f"{KNOWN}; not a usable MODEL"
        )
    index_path = os.path.join(root, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        print("  index: MISSING")
        try:
            names = sorted(
                name for name in os.listdir(root) if name.endswith(".safetensors")
            )
        except OSError as exc:
            print(f"  shards: cannot list directory ({exc})")
            return
        if not names:
            print("  shards: no safetensors files")
            return
        for name in names:
            size, err = file_size(os.path.join(root, name))
            if err:
                print(f"  shard {name}: UNREADABLE ({err})")
            elif size < MIN_SHARD:
                print(f"  shard {name}: TRIVIAL size={size}")
            else:
                print(f"  shard {name}: present size={size}")
        return

    size, err = file_size(index_path)
    if err:
        print(f"  index: UNREADABLE ({err})")
        return
    if size < MIN_INDEX:
        print(f"  index: TRIVIAL size={size}")
        return
    try:
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"  index: INVALID size={size} ({exc})")
        return
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        print(f"  index: EMPTY size={size}")
        return

    shards = []
    seen_shards = set()
    bad_names = 0
    for tensor, shard in weight_map.items():
        if (
            not isinstance(shard, str)
            or os.path.basename(shard) != shard
            or shard in ("", ".", "..")
        ):
            bad_names += 1
            print(f"  shard weight_map[{tensor!r}]: UNSAFE_NAME {shard!r}")
            continue
        if shard not in seen_shards:
            seen_shards.add(shard)
            shards.append(shard)
    present = nontrivial = missing = trivial = 0
    problems = []
    for shard in shards:
        path = os.path.join(root, shard)
        if not os.path.isfile(path):
            missing += 1
            problems.append(f"shard {shard}: MISSING")
            continue
        shard_size, shard_err = file_size(path)
        if shard_err:
            problems.append(f"shard {shard}: UNREADABLE ({shard_err})")
            continue
        present += 1
        if shard_size < MIN_SHARD:
            trivial += 1
            problems.append(f"shard {shard}: TRIVIAL size={shard_size}")
        else:
            nontrivial += 1
    print(
        f"  index: present size={size} tensors={len(weight_map)} "
        f"referenced_shards={len(shards)}"
    )
    print(
        "  shards: "
        f"referenced={len(shards)} present={present} nontrivial={nontrivial} "
        f"missing={missing} trivial={trivial} unsafe_names={bad_names}"
    )
    for line in problems:
        print(f"  {line}")


if not candidates:
    print("MODEL: no Qwen3.5-27B directory on the standard DLC mounts")
else:
    for cand in candidates:
        report(cand)
print(
    "NOTE: model completeness is not required; the GPU job may download "
    "Qwen3.5-27B onto node-local storage."
)
PY
model_rc=$?
if [[ "$model_rc" -ne 0 ]]; then
  bad "model candidate report failed"
fi

exit "$fail"

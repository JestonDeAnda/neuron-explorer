#!/usr/bin/env bash
# Depth survey for a list of layers: one short stats pass covering them all, then a random-subset sweep per layer.
#     setsid nohup scripts/run_depth.sh "0 4 9 13" 384 64 </dev/null >/dev/null 2>&1 &
set -uo pipefail
LAYERS="$1"; NNEUR="${2:-384}"; NCTRL="${3:-64}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source /workspace/env.sh
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p stats runs
need=""; for l in $LAYERS; do [ -f "stats/q35-4b_L$l.npz" ] || need="$need,$l"; done
if [ -n "$need" ]; then
  uv run python scripts/neuron_stats.py --layers "${need#,}" --tokens 200000 --out stats/q35-4b > depth_stats.log 2>&1
fi
for l in $LAYERS; do
  echo "L$l start $(date)" >> depth.log
  uv run python scripts/steer_sweep.py --stats "stats/q35-4b_L$l.npz" --design main_rel --dose-unit rel --neurons "random:$NNEUR" \
    --random-controls "$NCTRL" --max-num-seqs 320 --run "q35-4b-L$l-r$NNEUR" > "sweep_L$l-r$NNEUR.log" 2>&1 \
    && echo "L$l done $(date)" >> depth.log || echo "L$l FAILED $(date)" >> depth.log
done
echo ALL_DONE >> depth.log

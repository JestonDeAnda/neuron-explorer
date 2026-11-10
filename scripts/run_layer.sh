#!/usr/bin/env bash
# Stats pass then a full-layer sweep for one layer. Logs to stats_L<layer>.log and sweep_<run>.log; detach with
#     setsid nohup scripts/run_layer.sh 8 q35-4b-L8 </dev/null >/dev/null 2>&1 &
# Extra arguments pass through to scripts/steer_sweep.py.
set -euo pipefail
LAYER="$1"; RUN="$2"; shift 2
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source /workspace/env.sh
# shellcheck disable=SC1091
source .venv/bin/activate
mkdir -p stats runs
[ -f "stats/q35-4b_L${LAYER}.npz" ] || uv run python scripts/neuron_stats.py --layers "$LAYER" --tokens 2000000 \
    --out stats/q35-4b > "stats_L${LAYER}.log" 2>&1
uv run python scripts/steer_sweep.py --stats "stats/q35-4b_L${LAYER}.npz" --design main_rel --dose-unit rel --neurons all \
    --random-controls 256 --max-num-seqs 320 --run "$RUN" "$@" > "sweep_${RUN}.log" 2>&1
echo SWEEP_DONE >> "sweep_${RUN}.log"

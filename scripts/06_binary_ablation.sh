#!/bin/bash
# Ablation: RL v2 with a binary reward (exec_bin). Same SFT start, data, settings and seeds as RL v2; the only change
# is r = 1 if the formula keeps the interface and passes every solver test, else 0 (RL v2: 0.3 + 0.7 x share passed).
# The no-signal early stop is off: RL v2 never triggered it, but a sparser binary reward could stop a run that would
# otherwise learn. Resume-safe: finished steps are skipped; each step is retried on failure.
set -u; source "$(dirname "$0")/env.sh"
H=src/headroom
retry() {  # retry <tries> <command...>
  local n=$1; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "[$(date '+%F %T')] attempt $i/$n failed: $*" >&2; sleep 30
  done
  echo "[$(date '+%F %T')] GAVE UP: $*" >&2; return 1
}
for s in 42 43 44; do
  if [ ! -d models/rl_bin_s$s/final ]; then
    echo "[$(date '+%F %T')] train seed $s"
    rm -rf models/rl_bin_s$s   # a partial run restarts cleanly from the SFT adapter
    retry 3 python src/training/train_exec_grpo.py --model models/sft789/final --data data/rl/rl_v2/pool.jsonl \
      --reward exec_bin --max-steps 1000 --seed $s --stop-if-no-signal-after 0 --output-dir models/rl_bin_s$s || exit 1
  fi
done
for s in 42 43 44; do
  echo "[$(date '+%F %T')] evaluate seed $s"
  retry 3 python $H/run_formula.py models/rl_bin_s$s/final rl789_bin_s$s || exit 1
done
retry 3 python $H/verify_equivalence.py rl789_bin_s42 rl789_bin_s43 rl789_bin_s44 || exit 1
echo "[$(date '+%F %T')] BINARY_ABLATION_DONE"

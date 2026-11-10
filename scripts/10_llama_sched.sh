#!/bin/bash
# Llama-3.1-8B Stage II rerun with the schedule-aware, multi-date reward (exec_sched, data/rl/rl_v3), so all model
# families use the same final recipe. Reuses the Llama SFT adapter from 07_llama_baseline.sh. Resume-safe, retried.
set -u; source "$(dirname "$0")/env.sh"
H=src/headroom
M=meta-llama/Llama-3.1-8B-Instruct
retry() { local n=$1; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "[$(date '+%F %T')] attempt $i/$n failed: $*" >&2; sleep 30; done; echo "[$(date '+%F %T')] GAVE UP: $*" >&2; return 1; }
for s in 42 43 44; do
  if [ ! -d models/llama_rl_sched_s$s/final ]; then
    echo "[$(date '+%F %T')] train seed $s"
    rm -rf models/llama_rl_sched_s$s
    retry 3 python src/training/train_exec_grpo.py --model models/llama_sft789/final --data data/rl/rl_v3/pool.jsonl \
      --reward exec_sched --max-steps 1000 --seed $s --binder-model $M --output-dir models/llama_rl_sched_s$s || exit 1
  fi
done
for s in 42 43 44; do
  echo "[$(date '+%F %T')] evaluate seed $s"
  retry 3 python $H/run_formula.py models/llama_rl_sched_s$s/final llama_rl_sched_s$s || exit 1
done
retry 3 python $H/verify_equivalence.py llama_rl_sched_s42 llama_rl_sched_s43 llama_rl_sched_s44 || exit 1
echo "[$(date '+%F %T')] LLAMA_SCHED_DONE"

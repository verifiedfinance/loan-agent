#!/bin/bash
# Reward v3: Stage II with the schedule- and condition-aware reward (exec_sched) on the 828-clause pool
# (data/rl/rl_v3: the 586 v2 clauses unchanged + 242 with date schedules or condition flags).
# Everything else as RL v2 (04_train_rl.sh): same SFT start, 1,000 steps, seeds 42/43/44. Resume-safe, retried.
set -u; source "$(dirname "$0")/env.sh"
H=src/headroom
retry() { local n=$1; shift; for i in $(seq 1 "$n"); do "$@" && return 0; echo "[$(date '+%F %T')] attempt $i/$n failed: $*" >&2; sleep 30; done; echo "[$(date '+%F %T')] GAVE UP: $*" >&2; return 1; }
for s in 42 43 44; do
  if [ ! -d models/rl_sched_s$s/final ]; then
    echo "[$(date '+%F %T')] train seed $s"
    rm -rf models/rl_sched_s$s
    retry 3 python src/training/train_exec_grpo.py --model models/sft789/final --data data/rl/rl_v3/pool.jsonl \
      --reward exec_sched --max-steps 1000 --seed $s --output-dir models/rl_sched_s$s || exit 1
  fi
done
for s in 42 43 44; do
  echo "[$(date '+%F %T')] evaluate seed $s"
  retry 3 python $H/run_formula.py models/rl_sched_s$s/final rl789_sched_s$s || exit 1
done
retry 3 python $H/verify_equivalence.py rl789_sched_s42 rl789_sched_s43 rl789_sched_s44 || exit 1
echo "[$(date '+%F %T')] SCHEDULE_REWARD_DONE"

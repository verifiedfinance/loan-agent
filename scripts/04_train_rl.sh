#!/bin/bash
# Stage 4: GRPO from the SFT adapter on unseen unlabeled clauses.
#   RL v2 (the paper's model): headroom prompt format + reward exec_sig (0 unless the formula keeps exactly the required
#   parameters, else 0.3 + 0.7 x solver-checked execution score), seeds 42/43/44.
#   RL v1 (reported as a failure): old prompt format + reward exec, seed 42.
set -eu; source "$(dirname "$0")/env.sh"
python src/headroom/rl_v2_pool.py read                 # Claude names each training clause's line items
python src/headroom/rl_v2_pool.py build                # -> data/rl/rl_v2/pool.jsonl (no schedules / conditions)
for s in 42 43 44; do
  [ -d models/rl_v2_s$s/final ] || python src/training/train_exec_grpo.py --model models/sft789/final \
    --data data/rl/rl_v2/pool.jsonl --reward exec_sig --max-steps 1000 --seed $s --output-dir models/rl_v2_s$s
done
[ -d models/rl_v1/final ] || python src/training/train_exec_grpo.py --model models/sft789/final \
  --data data/rl/pool_fresh.jsonl --reward exec --max-steps 1000 --seed 42 --output-dir models/rl_v1

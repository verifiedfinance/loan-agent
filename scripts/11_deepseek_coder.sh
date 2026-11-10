#!/bin/bash
# Fourth model family: the same experiments with DeepSeek-Coder-6.7B-Instruct (bf16), a second code model from a
# different lab, in place of Qwen2.5-Coder-7B-Instruct.
#   reader      : reads clause + certificate, 1,024 tokens, cut-off cases rerun with 4,096 (as for the other readers)
#   formalizer  : prompted; SFT on the same 789 pairs; GRPO with the final schedule-aware reward (exec_sched,
#                 data/rl/rl_v3), 1,000 steps, seeds 42/43/44
# Every setting is copied from 08_mistral_baseline.sh; only the base model changes. DeepSeek-Coder's tokenizer is loaded
# through lora_utils.load_tokenizer, which keeps spaces and newlines intact under transformers 5.
# Resume-safe: finished steps are skipped; each step is retried on failure.
set -u; source "$(dirname "$0")/env.sh"
H=src/headroom
M=deepseek-ai/deepseek-coder-6.7b-instruct
retry() {  # retry <tries> <command...>
  local n=$1; shift
  for i in $(seq 1 "$n"); do
    "$@" && return 0
    echo "[$(date '+%F %T')] attempt $i/$n failed: $*" >&2; sleep 30
  done
  echo "[$(date '+%F %T')] GAVE UP: $*" >&2; return 1
}
step() { echo "[$(date '+%F %T')] $*"; }

# 1. reader
step "reader"
retry 3 python $H/run_direct.py hf $M dscoder7b || exit 1
retry 3 python $H/rerun_truncated.py $M dscoder7b || exit 1

# 2. formalizer: prompted, then SFT, then GRPO x 3 seeds
step "prompted formalizer"
retry 3 python $H/run_formula.py $M dscoder7b_prompted || exit 1
if [ ! -d models/dscoder_sft789/final ]; then
  step "SFT"
  rm -rf models/dscoder_sft789
  retry 3 python src/training/train_sft.py --model $M --lora-rank 16 --learning-rate 1e-4 \
    --num-train-epochs 1 --data data/sft/sft_pairs.jsonl --output-dir models/dscoder_sft789 || exit 1
fi
step "SFT formalizer"
retry 3 python $H/run_formula.py models/dscoder_sft789/final dscoder_sft789 || exit 1
for s in 42 43 44; do
  if [ ! -d models/dscoder_rl_sched_s$s/final ]; then
    step "GRPO seed $s"
    rm -rf models/dscoder_rl_sched_s$s
    retry 3 python src/training/train_exec_grpo.py --model models/dscoder_sft789/final --data data/rl/rl_v3/pool.jsonl \
      --reward exec_sched --max-steps 1000 --seed $s --binder-model $M --output-dir models/dscoder_rl_sched_s$s || exit 1
  fi
  step "evaluate seed $s"
  retry 3 python $H/run_formula.py models/dscoder_rl_sched_s$s/final dscoder_rl_sched_s$s || exit 1
done

# 3. verify the formulas
step "verify"
retry 3 python $H/verify_equivalence.py dscoder7b_prompted dscoder_sft789 dscoder_rl_sched_s42 dscoder_rl_sched_s43 \
  dscoder_rl_sched_s44 || exit 1
step "DSCODER_BASELINE_DONE"

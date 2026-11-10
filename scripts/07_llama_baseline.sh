#!/bin/bash
# Second model family: the same experiments with Llama-3.1-8B-Instruct (bf16) in place of Qwen2.5-Coder-7B-Instruct.
#   reader      : reads clause + certificate, 1,024 tokens, cut-off cases rerun with 4,096 (as for the Qwen readers)
#   formalizer  : prompted; SFT on the same 789 pairs; GRPO with the same reward (exec_sig), 1,000 steps, seeds 42/43/44
# Every setting is copied from 02_train_sft.sh, 04_train_rl.sh and 05_evaluate.sh; only the base model changes.
# Resume-safe: finished steps are skipped; each step is retried on failure.
set -u; source "$(dirname "$0")/env.sh"
H=src/headroom
M=meta-llama/Llama-3.1-8B-Instruct
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
retry 3 python $H/run_direct.py hf $M llama8b || exit 1
retry 3 python $H/rerun_truncated.py $M llama8b || exit 1

# 2. formalizer: prompted, then SFT, then GRPO x 3 seeds
step "prompted formalizer"
retry 3 python $H/run_formula.py $M llama8b_prompted || exit 1
if [ ! -d models/llama_sft789/final ]; then
  step "SFT"
  rm -rf models/llama_sft789
  retry 3 python src/training/train_sft.py --model $M --lora-rank 16 --learning-rate 1e-4 \
    --num-train-epochs 1 --data data/sft/sft_pairs.jsonl --output-dir models/llama_sft789 || exit 1
fi
for s in 42 43 44; do
  if [ ! -d models/llama_rl_v2_s$s/final ]; then
    step "GRPO seed $s"
    rm -rf models/llama_rl_v2_s$s
    retry 3 python src/training/train_exec_grpo.py --model models/llama_sft789/final --data data/rl/rl_v2/pool.jsonl \
      --reward exec_sig --max-steps 1000 --seed $s --binder-model $M --output-dir models/llama_rl_v2_s$s || exit 1
  fi
done

# 3. evaluate the formalizers and verify their formulas
step "evaluate"
retry 3 python $H/run_formula.py models/llama_sft789/final llama_sft789 || exit 1
for s in 42 43 44; do retry 3 python $H/run_formula.py models/llama_rl_v2_s$s/final llama_rl_v2_s$s || exit 1; done
retry 3 python $H/verify_equivalence.py llama8b_prompted llama_sft789 llama_rl_v2_s42 llama_rl_v2_s43 llama_rl_v2_s44 || exit 1
step "LLAMA_BASELINE_DONE"

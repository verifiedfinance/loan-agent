#!/bin/bash
# Second model family: the same experiments with Mistral-7B-Instruct-v0.3 (bf16) in place of Qwen2.5-Coder-7B-Instruct.
#   reader      : reads clause + certificate, 1,024 tokens, cut-off cases rerun with 4,096 (as for the Qwen readers)
#   formalizer  : prompted; SFT on the same 789 pairs; GRPO with the same reward (exec_sig), 1,000 steps, seeds 42/43/44
# Every setting is copied from 02_train_sft.sh, 04_train_rl.sh and 05_evaluate.sh; only the base model changes.
# Resume-safe: finished steps are skipped; each step is retried on failure.
set -u; source "$(dirname "$0")/env.sh"
H=src/headroom
M=mistralai/Mistral-7B-Instruct-v0.3
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
retry 3 python $H/run_direct.py hf $M mistral7b || exit 1
retry 3 python $H/rerun_truncated.py $M mistral7b || exit 1

# 2. formalizer: prompted, then SFT, then GRPO x 3 seeds
step "prompted formalizer"
retry 3 python $H/run_formula.py $M mistral7b_prompted || exit 1
if [ ! -d models/mistral_sft789/final ]; then
  step "SFT"
  rm -rf models/mistral_sft789
  retry 3 python src/training/train_sft.py --model $M --lora-rank 16 --learning-rate 1e-4 \
    --num-train-epochs 1 --data data/sft/sft_pairs.jsonl --output-dir models/mistral_sft789 || exit 1
fi
# The Stage II reward (exec_sig = paper's FormaFin, or exec_sched = schedule-aware) is decided after the Qwen
# schedule-reward run (09); the decision is written to data/rl/mistral_reward. Wait for it at most 3 hours, then use
# exec_sig so the GPU never sits idle.
CHOICE=data/rl/mistral_reward
for i in $(seq 1 180); do [ -s $CHOICE ] && break; sleep 60; done
REWARD=$(cat $CHOICE 2>/dev/null || echo exec_sig)
case $REWARD in exec_sched) POOL=data/rl/rl_v3/pool.jsonl ;; *) REWARD=exec_sig; POOL=data/rl/rl_v2/pool.jsonl ;; esac
step "Stage II reward: $REWARD ($POOL)"
for s in 42 43 44; do
  if [ ! -d models/mistral_rl_v2_s$s/final ]; then
    step "GRPO seed $s"
    rm -rf models/mistral_rl_v2_s$s
    retry 3 python src/training/train_exec_grpo.py --model models/mistral_sft789/final --data $POOL \
      --reward $REWARD --max-steps 1000 --seed $s --binder-model $M --output-dir models/mistral_rl_v2_s$s || exit 1
  fi
done

# 3. evaluate the formalizers and verify their formulas
step "evaluate"
retry 3 python $H/run_formula.py models/mistral_sft789/final mistral_sft789 || exit 1
for s in 42 43 44; do retry 3 python $H/run_formula.py models/mistral_rl_v2_s$s/final mistral_rl_v2_s$s || exit 1; done
retry 3 python $H/verify_equivalence.py mistral7b_prompted mistral_sft789 mistral_rl_v2_s42 mistral_rl_v2_s43 mistral_rl_v2_s44 || exit 1
step "MISTRAL_BASELINE_DONE"

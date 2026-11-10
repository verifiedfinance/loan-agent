#!/bin/bash
# Benchmark score along Stage II: evaluate the GRPO checkpoints at steps 250 / 500 / 750 for all three model families and
# seeds (step 0 = SFT and step 1000 = final are already scored). 27 formalizer runs, three at a time on one GPU.
# Output: results/headroom/answers/formula_ckpt_<family>_s<seed>_<step>.jsonl (resume-safe: finished runs are skipped).
set -u; source "$(dirname "$0")/env.sh"
jobs=()
for fam in qwen:rl_sched_s llama:llama_rl_sched_s mistral:mistral_rl_v2_s; do
  name=${fam%%:*}; dir=${fam#*:}
  for s in 42 43 44; do for step in 250 500 750; do
    jobs+=("models/${dir}${s}/checkpoint-${step} ckpt_${name}_s${s}_${step}")
  done; done
done
run_one() {
  set -- $1
  out=results/headroom/answers/formula_$2.jsonl
  if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 1064 ]; then echo "SKIP $2"; return 0; fi
  for i in 1 2 3; do
    log=$(python src/headroom/run_formula.py "$1" "$2" 2>&1)
    echo "$log" | grep -E "FORMULA_DONE|Traceback" | tail -2
    echo "$log" | grep -q FORMULA_DONE && return 0
    echo "[$(date '+%F %T')] attempt $i failed: $2"; sleep 30
  done
  echo "[$(date '+%F %T')] GAVE UP: $2"
}
export -f run_one
echo "[$(date '+%F %T')] start: ${#jobs[@]} checkpoint evaluations, 3 in parallel"
printf '%s\n' "${jobs[@]}" | xargs -P 3 -I{} bash -c 'run_one "{}"'
echo "[$(date '+%F %T')] CKPT_EVAL_DONE"

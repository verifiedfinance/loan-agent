#!/bin/bash
# Large reader baseline: Llama-3.3-70B-Instruct reads clause + certificate and answers directly (no formalizer, no training).
# Same protocol as every other reader (05_evaluate.sh, 07_llama_baseline.sh): greedy, 1,024 answer tokens, cut-off
# cases rerun with 4,096. Only difference: the weights are loaded in 4-bit NF4 (bitsandbytes, hf_generation.FOUR_BIT)
# because bf16 needs ~140 GB and the GPU has 96 GB. Resume-safe; on failure (e.g. out of memory on the longest
# prompts) each attempt halves the batch: 16 -> 8 -> 4.
set -u; source "$(dirname "$0")/env.sh"
H=src/headroom
M=meta-llama/Llama-3.3-70B-Instruct
step() { echo "[$(date '+%F %T')] $*"; }
shrink() {  # shrink <command...>: run with READER_BATCH 16, 8, 4 until it succeeds
  for b in 16 8 4; do
    READER_BATCH=$b "$@" && return 0
    echo "[$(date '+%F %T')] attempt with batch $b failed: $*" >&2; sleep 30
  done
  echo "[$(date '+%F %T')] GAVE UP: $*" >&2; return 1
}

step "download check"
python -c "from huggingface_hub import snapshot_download as s; s('$M', allow_patterns=['*.safetensors','*.json','tokenizer*'])" || exit 1
step "reader (1,024 tokens)"
shrink python $H/run_direct.py hf $M llama70b || exit 1
step "rerun cut-off answers (4,096 tokens)"
shrink python $H/rerun_truncated.py $M llama70b || exit 1
step "LLAMA70B_READER_DONE"

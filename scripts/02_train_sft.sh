#!/bin/bash
# Stage 2: teacher-written covenant -> SMT pairs (Claude CLI, Z3- and judge-filtered), then LoRA SFT of Qwen2.5-Coder-7B.
set -eu; source "$(dirname "$0")/env.sh"
python src/training/make_sft_pairs.py                  # -> data/sft/sft_pairs.jsonl (789 pairs)
[ -d models/sft789/final ] || python src/training/train_sft.py --model Qwen/Qwen2.5-Coder-7B-Instruct --lora-rank 16 --learning-rate 1e-4 \
  --num-train-epochs 1 --data data/sft/sft_pairs.jsonl --output-dir models/sft789

#!/bin/bash
# Stage 5: both arms on the 1,064 cases, sensitivity reruns, formal verification, scores (resume-safe).
set -eu; source "$(dirname "$0")/env.sh"
H=src/headroom
# formula arm: the model writes covenant_holds over the certificate's parameters; Z3 answers
python $H/run_formula.py Qwen/Qwen2.5-Coder-7B-Instruct qwen7b_prompted
python $H/run_formula.py models/sft789/final sft789
python $H/run_formula.py models/rl_v1/final rl789_s42
python $H/run_formula.py models/rl_v2_s42/final rl789_v2
python $H/run_formula.py models/rl_v2_s43/final rl789_v2_s43
python $H/run_formula.py models/rl_v2_s44/final rl789_v2_s44
for n in qwen7b_prompted:Qwen/Qwen2.5-Coder-7B-Instruct sft789:models/sft789/final rl789_s42:models/rl_v1/final \
         rl789_v2:models/rl_v2_s42/final rl789_v2_s43:models/rl_v2_s43/final rl789_v2_s44:models/rl_v2_s44/final; do
  python $H/run_formula.py ${n#*:} ${n%%:*} --lenient    # sensitivity: typed subset of parameters
done
# direct arm: the model reads clause + certificate and answers
python $H/run_direct.py claude claude-sonnet-5-5 claude_sonnet55
python $H/run_direct.py hf Qwen/Qwen2.5-Coder-7B-Instruct qwen7b
python $H/run_direct.py hf Qwen/Qwen2.5-Coder-32B-Instruct qwen32b
python $H/run_direct.py hf Qwen/Qwen3.8-27B qwen27b
for m in Qwen/Qwen2.5-Coder-7B-Instruct:qwen7b Qwen/Qwen2.5-Coder-32B-Instruct:qwen32b Qwen/Qwen3.8-27B:qwen27b; do
  python $H/rerun_truncated.py ${m%:*} ${m##*:}         # cut-off answers regenerated with 4,096 tokens
done
python $H/verify_equivalence.py qwen7b_prompted sft789 rl789_s42 rl789_v2 rl789_v2_s43 rl789_v2_s44
python $H/score_headroom.py                            # -> results/headroom/{summary.json,results.md}

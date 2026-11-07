#!/bin/bash
# Stage 3: the headroom benchmark. Two independent readings per clause (Claude; Qwen3.8-27B on the GPU), kept only when
# they agree; constructed certificates; the reference is validated through the formula-arm Z3 code (must be 100%).
set -eu; source "$(dirname "$0")/env.sh"
export JUDGE_BACKEND=claude_cli JUDGE_MODEL=claude-sonnet-5-5
python src/headroom/build_specs.py read_a              # -> data/headroom/reader_a.jsonl
python src/headroom/build_specs.py read_b              # -> data/headroom/reader_b.jsonl
python src/headroom/build_specs.py merge               # -> data/headroom/specs.jsonl, specs_summary.json
python src/headroom/make_cases.py                      # -> data/headroom/cases.jsonl, validation.json
python -c "import json; v=json.load(open('data/headroom/validation.json')); assert v['validation_correct']==v['cases'], v"

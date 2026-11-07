# Sourced by every stage script: repository root, Python environment, src/ on PYTHONPATH, pinned Codex CLI.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"   # bash or zsh
[ -d "$ROOT/.venv/bin" ] && export PATH="$ROOT/.venv/bin:$PATH"   # else the active environment (requirements.lock.txt)
[ -d "$ROOT/bin/codex-cli" ] && export PATH="$ROOT/bin/codex-cli:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

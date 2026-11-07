#!/bin/bash
# Stage 6 (CPU): paper tables (LaTeX + Markdown) and figures (PDF + PNG), then the notebook end to end.
set -eu; source "$(dirname "$0")/env.sh"
python src/headroom/paper.py                           # -> results/paper/{tables,figures}
python scripts/build_notebook.py                       # -> notebooks/headroom_paper.ipynb, executed

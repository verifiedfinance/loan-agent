"""Current formalization protocol: pointwise covenant_holds functions."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
from pointwise import (
    PROTOCOL, CONVENTION, SYSTEM, TEACHER, JUDGE, source_payload, student_prompt,
    validate_smt, judge_accepts, commands,
)

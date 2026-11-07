"""One shared parameter-comment convention for the teacher, student and judge."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import re

VERSION = 'parameter_comments_v1'
RULES = '''
Parameter meanings are part of the formal specification. Document EVERY parameter
with a precise source-grounded SMT comment. Use either a named comment before the
function, "; parameter_name: meaning", or a comment directly attached to that
parameter's declaration, "(parameter_name Real_or_Bool) ; meaning". Both styles are
equally valid. Do not reject clear parameter meanings merely because of comment placement.
For Real inputs, identify the quantity and, where stated, entity, units and measurement
period. For Boolean inputs, state the exact date/event/obligation condition and relevant
exceptions. For example, name the obligation types excluded from an unpaid-obligation
flag; "including all stated exceptions" is insufficient. Meanings must follow the
supplied source and declared assumptions, without inventing definitions or financial
positivity. A named ratio may remain an atomic input when its arithmetic definition is
unavailable under the existing task convention.
Comments define observations supplied to the function; they do not enforce constraints.
Preserve numerical constraints, explicit applicability relationships and temporal scope
in the formula. Do not hide the entire covenant inside an opaque covenant_active input
or use comments to claim that omitted formula conditions are enforced. Comments and
formula must agree. Keep each description concise and specific. This adds no new SMT
constructs and does not change the pointwise temporal convention.
Underspecified or contradictory parameter meanings cannot establish faithful formalization.
'''


def with_parameter_semantics(prompt):
    return prompt + '\n' + RULES


def student_prompt(record):
    from autoformalization import student_prompt as original_prompt
    messages = original_prompt(record)
    messages[0]['content'] = with_parameter_semantics(messages[0]['content'])
    return messages


def comment_coverage(smt, parameters):
    """Check unambiguously associated comments; source grounding requires review."""
    found = {}
    named = re.findall(r'^[ \t]*;[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*:[ \t]*([^\r\n]+)', smt, re.M)
    inline = re.findall(r'\(([A-Za-z_][A-Za-z0-9_]*)[ \t]+(?:Real|Bool)\)[ \t]*;[ \t]*([^\r\n]+)', smt)
    for name, meaning in named + inline:
        if meaning.strip():
            found.setdefault(name, []).append(meaning.strip())
    missing = sorted(name for name in parameters if name not in found)
    duplicate = sorted(name for name in parameters if len(found.get(name, [])) > 1)
    return {'complete': bool(parameters) and not missing and not duplicate,
            'missing_parameters': missing, 'duplicate_parameters': duplicate,
            'meanings': {name: found[name][0] for name in parameters if name in found},
            'limitation': 'Comment presence is not evidence of correct source grounding.'}

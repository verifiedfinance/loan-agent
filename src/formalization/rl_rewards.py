"""Online GRPO reward: SMT validity plus separate LLM semantic correspondence."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
from functools import lru_cache
import json
import logging
from autoformalization import JUDGE, source_payload, validate_smt, judge_accepts

DENSE_RUBRIC = '''
Also return semantic_score, an integer from 0 to 3 when the source is sufficient:
0: Fundamentally incorrect formalization.
1: Multiple important semantic errors.
2: Mostly correct with a localized material error.
3: Fully semantically faithful.
Grade the material effect of the errors, not the number of issues. One serious
error can be worse than several minor ones. Explain the grade in reason using
the source and formula. Preserve all existing binary verdict fields and rules.
Score 3 if and only if equivalent, complete, and no_unsupported_assumptions are
true, needs_context is false, and issues is empty. If source meaning is missing,
set needs_context=true and semantic_score=null; do not guess or give a grade.
'''


class JudgeUnavailableError(RuntimeError):
    """No usable semantic feedback; the reward batch must not update the model."""


class InsufficientContextError(JudgeUnavailableError):
    """A source abstention, kept separate from transport/schema failures."""
    def __init__(self, verdict):
        super().__init__('Insufficient context')
        self.verdict = verdict


def checked_verdict(verdict, *, dense=False):
    """A malformed verdict is unavailable feedback, not a semantic rejection."""
    flags = ('equivalent', 'complete', 'no_unsupported_assumptions', 'needs_context')
    if (not isinstance(verdict, dict)
            or any(type(verdict.get(key)) is not bool for key in flags)
            or not isinstance(verdict.get('issues'), list)
            or any(not isinstance(issue, str) for issue in verdict['issues'])
            or not isinstance(verdict.get('reason'), str) or not verdict['reason'].strip()):
        raise JudgeUnavailableError('Malformed semantic judge verdict')
    if dense:
        if 'semantic_score' not in verdict:
            raise JudgeUnavailableError('Missing semantic score')
        score = verdict['semantic_score']
        if verdict['needs_context']:
            if score is not None:
                raise JudgeUnavailableError('Missing-context verdict requires a null semantic score')
        else:
            if type(score) is not int or score not in (0, 1, 2, 3):
                raise JudgeUnavailableError('Invalid semantic score')
            if (score == 3) != judge_accepts(verdict):
                raise JudgeUnavailableError('Inconsistent semantic verdict')
    return verdict


def score_autoformalization(completion, record, semantic_judge, dense=False):
    if isinstance(completion, list):
        completion = completion[0].get('content', '') if len(completion) == 1 and isinstance(completion[0], dict) else ''
    validity = validate_smt(completion)
    if not validity['valid']:
        return {'reward': 0.0, 'validity': validity, 'semantic': None}
    verdict = checked_verdict(semantic_judge(record, completion), dense=dense)
    if dense and verdict['needs_context']:
        raise InsufficientContextError(verdict)
    reward = (0.2 + 0.8 * (verdict['semantic_score'] / 3) if dense
              else float(judge_accepts(verdict)))
    return {'reward': reward, 'validity': validity, 'semantic': verdict}


def make_grpo_reward(client, audit_callback=None, *, max_attempts=2,
                     cache_size=1024, judge_version=None, judge_prompt=None, dense=False):
    """Use an isolated judge client; retry failures, then abort the reward batch.

    Successful judgments (including rejections) are cached in memory per reward
    instance. Keys include the full source, SMT, protocol/convention, judge prompt,
    and model/version/configuration. Supply judge_version for a custom client
    without model metadata. No failed judgment is cached or given a numeric reward.
    """
    from verifiqa.generation.llm_client import ChatMessage
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('max_attempts must be a positive integer')
    if type(cache_size) is not int or cache_size < 0:
        raise ValueError('cache_size must be a nonnegative integer')
    if not (judge_version or getattr(client, 'model', None)):
        raise ValueError('Provide judge_version when the client has no model name')

    @lru_cache(maxsize=cache_size)
    def judge_response(payload, judge_prompt, judge_identity):
        for attempt in range(1, max_attempts + 1):
            try:
                answer = client.chat([ChatMessage('system', judge_prompt),
                                      ChatMessage('user', payload)], stage='grpo_semantic_reward')
                verdict = checked_verdict(json.loads(answer), dense=dense)
                # Cache serialized data so audit callbacks cannot mutate cached verdicts.
                return json.dumps(verdict)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    'Semantic judge attempt %d/%d failed (%s)', attempt, max_attempts, type(exc).__name__)
                if audit_callback:
                    audit_callback({'status': 'judge_failure', 'reward': None,
                                    'attempt': attempt, 'error_type': type(exc).__name__})
                if attempt == max_attempts:
                    raise JudgeUnavailableError(
                        f'Semantic judge unavailable after {max_attempts} attempts; reward batch aborted') from exc

    def semantic_judge(record, completion):
        payload = json.dumps({**source_payload(record), 'candidate_smt2': completion}, sort_keys=True)
        identity = json.dumps({'model': getattr(client, 'model', None), 'version': judge_version,
                               'configuration': getattr(client, 'run_metadata', {}), 'dense': dense}, sort_keys=True)
        return json.loads(judge_response(payload, JUDGE if judge_prompt is None else judge_prompt, identity))

    def autoformalization_reward(completions, text, declared_assumptions=None,
                                source_context=None, **kwargs):
        if len(completions) != len(text):
            raise ValueError('Each completion requires its source covenant')
        assumptions = declared_assumptions if declared_assumptions is not None else [[] for _ in text]
        contexts = source_context if source_context is not None else [[] for _ in text]
        if len(assumptions) != len(text) or len(contexts) != len(text):
            raise ValueError('Assumptions and source context must align with completions')
        results = [score_autoformalization(c, {'text': t, 'declared_assumptions': a,
                                              'source_context': context}, semantic_judge, dense=dense)
                   for c, t, a, context in zip(completions, text, assumptions, contexts)]
        if audit_callback:
            for result in results:
                audit_callback(result)
        return [result['reward'] for result in results]
    return autoformalization_reward

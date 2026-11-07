#!/usr/bin/env python3
"""Single-GPU TRL GRPO using the existing Z3 and semantic reward."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent


def training_indices(records, heldout):
    """Exclude held-out contracts and their exact-duplicate-connected contracts."""
    normalize = lambda row: ' '.join(row['text'].casefold().split())
    contracts = {row['contract_id'] for row in heldout}
    texts = {normalize(row) for row in heldout}
    while True:
        previous = (len(contracts), len(texts))
        for row in records:
            text = normalize(row)
            if row['contract_id'] in contracts or text in texts:
                contracts.add(row['contract_id'])
                texts.add(text)
        if previous == (len(contracts), len(texts)):
            break
    ids = {row['id'] for row in heldout}
    return [i for i, row in enumerate(records)
            if row['id'] not in ids and row['contract_id'] not in contracts and normalize(row) not in texts]


def assert_matches_baseline(metadata, baseline):
    """Only reward behavior and output location may change in a matched experiment."""
    current = json.loads(json.dumps(metadata, default=str))
    fields = ('sft_checkpoint', 'source_sha256', 'evaluation_sha256', 'record_ids',
              'evaluation_record_ids', 'excluded_development_record_ids', 'versions', 'judge')
    differences = [field for field in fields if current[field] != baseline[field]]
    settings = lambda row: {k: v for k, v in row['training'].items() if k != 'output_dir'}
    a, b = settings(current), settings(baseline)
    differences += [f'training.{k}' for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
    if differences:
        raise ValueError('Baseline mismatch: ' + ', '.join(sorted(differences)))


def make_logged_reward(reward, output_dir, num_generations=2, phase_getter=lambda: 'train', ending_token_ids=None,
                       abstain_on_missing_context=False):
    """Log completions and within-prompt reward variation for this single-GPU run."""
    reward_log = output_dir / 'rewards.jsonl'
    group_log = output_dir / 'reward_groups.jsonl'
    reward_log.write_text('')
    group_log.write_text('')

    def logged_reward(completions, text, **kwargs):
        ids = kwargs['id']
        if len(ids) != len(completions) or any(record_id is None for record_id in ids):
            raise ValueError('Every completion needs its source record ID for group logging')
        from rl_rewards import InsufficientContextError
        abstention = None
        try:
            scores = reward(completions=completions, text=text, **kwargs)
        except InsufficientContextError as exc:
            # This pilot generates exactly one complete prompt group per call.
            # Withhold the whole source group, including already-scored candidates.
            if not abstain_on_missing_context or len(set(ids)) != 1 or len(ids) != num_generations:
                raise
            scores = [None] * len(completions)
            abstention = exc.verdict
        step = getattr(kwargs.get('trainer_state'), 'global_step', None)
        phase = phase_getter()
        token_ids = kwargs.get('completion_ids')
        masked = ([not tokens or tokens[-1] not in ending_token_ids for tokens in token_ids]
                  if token_ids is not None and ending_token_ids is not None else [None] * len(scores))
        groups = defaultdict(list)
        with reward_log.open('a') as handle:
            for record_id, completion, score, mask in zip(ids, completions, scores, masked):
                groups[record_id].append((score, mask))
                handle.write(json.dumps({'step': step, 'phase': phase, 'id': record_id, 'masked': mask,
                                         'completion': completion, 'reward': score}) + '\n')
        complete = [values for values in groups.values() if len(values) == num_generations]
        scored = [[(score, mask) for score, mask in values if score is not None] for values in complete]
        informative = sum(len({score for score, _ in values}) > 1 for values in scored)
        effective = sum(any(mask is False and abs(score - sum(s for s, _ in values) / len(values)) > 1e-8
                            for score, mask in values) for values in scored if values)
        summary = {'step': step, 'phase': phase, 'complete_groups': len(complete), 'informative_groups': informative,
                   'effective_informative_groups': effective if all(m is not None for m in masked) else None,
                   'groups_with_reward_one': sum(any(score == 1.0 for score, _ in values) for values in scored),
                   'semantic_reward_groups': sum({0.2, 1.0}.issubset({score for score, _ in values}) for values in scored),
                   'effective_semantic_reward_groups': sum(
                       {0.2, 1.0}.issubset({score for score, mask in values if mask is False}) for values in complete
                   ) if all(m is not None for m in masked) else None,
                   'graded_semantic_reward_groups': sum(len({s for s, _ in values if s > 0}) > 1 for values in scored),
                   'effective_graded_semantic_reward_groups': sum(
                       len({s for s, mask in values if s > 0 and mask is False}) > 1 for values in scored
                   ) if all(m is not None for m in masked) else None,
                   'unscored_groups': sum(not values for values in scored),
                   'unscored_completions': sum(s is None for s in scores),
                   'context_abstention': abstention,
                   'masked_completions': sum(m is True for m in masked),
                   'informative_fraction': informative / len(complete) if complete else None,
                   'incomplete_groups': len(groups) - len(complete)}
        with group_log.open('a') as handle:
            handle.write(json.dumps(summary) + '\n')
        print('Reward counts:', dict(Counter(scores)), flush=True)
        print(f'Informative GRPO groups: {informative}/{len(complete)} '
              f'(incomplete groups: {summary["incomplete_groups"]})', flush=True)
        return scores
    return logged_reward


def no_reward_signal(groups, minimum_groups):
    """Stop only after enough complete training groups with known, unusable advantages."""
    rows = [row for row in groups if row['phase'] == 'train']
    return (minimum_groups > 0 and sum(row['complete_groups'] for row in rows) >= minimum_groups
            and all(row['effective_informative_groups'] is not None for row in rows)
            and sum(row['effective_informative_groups'] for row in rows) == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=P.MODELS / 'sft789/final', help='Local SFT checkpoint')
    parser.add_argument('--max-steps', type=int, default=5)
    parser.add_argument('--learning-rate', type=float, default=5e-7)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--num-generations', type=int, default=2)
    parser.add_argument('--max-completion-length', type=int, default=512)
    parser.add_argument('--parameter-comments', action='store_true', help='Use the frozen parameter rules in student and judge prompts')
    parser.add_argument('--stop-if-no-signal-after', type=int, default=0,
                        help='Stop after this many complete groups if none has usable reward differences; 0 disables')
    parser.add_argument('--data', type=Path, default=P.POOL_RATIO)
    parser.add_argument('--train-limit', type=int, default=16, help='Training records after exclusions; 0 uses all')
    parser.add_argument('--eval-data', type=Path, help='Held-out source JSONL; related training contracts are excluded')
    parser.add_argument('--checkpoint-steps', type=int, default=25, help='Save and validate at this interval when --eval-data is set')
    parser.add_argument('--skip-trainer-eval', action='store_true', help='Use --eval-data for exclusions only; evaluate saved checkpoints separately')
    parser.add_argument('--output-dir', type=Path, default=P.MODELS / 'grpo_run')
    parser.add_argument('--validity-only-reward', action='store_true', help='Ablation: reward 1 for valid SMT, no judge')
    parser.add_argument('--mutation-reward', action='store_true', help='Apply the existing mutation-witness penalty')
    parser.add_argument('--dense-reward', action='store_true', help='Use the audited 0-3 semantic grades')
    parser.add_argument('--non-temporal-reward', action='store_true', help='Exclude temporal correctness from the dense judge')
    parser.add_argument('--baseline-config', type=Path, help='Require matching student, data, versions, and training settings')
    parser.add_argument('--check-data', action='store_true', help='Check the JSONL schema and exit without loading a model')
    args = parser.parse_args()
    if (args.max_steps < 1 or args.train_limit < 0 or args.checkpoint_steps < 1
            or args.learning_rate <= 0 or args.num_generations < 2 or args.max_completion_length < 1
            or args.stop_if_no_signal_after < 0 or args.gradient_accumulation_steps < 1):
        parser.error('Steps must be positive and --train-limit nonnegative')
    if args.dense_reward and args.mutation_reward:
        parser.error('Dense and mutation rewards cannot be combined')
    if args.non_temporal_reward and not args.dense_reward:
        parser.error('--non-temporal-reward requires --dense-reward')

    from datasets import load_dataset
    from autoformalization import student_prompt
    if args.parameter_comments:
        from parameter_semantics import student_prompt
    source = args.data
    data = load_dataset('json', data_files=str(source), split='train')
    if not len(data) or not {'prompt', 'text', 'id'}.issubset(data.column_names):
        raise ValueError('GRPO data requires nonempty id, prompt, and text columns')
    for row in data:
        if not isinstance(row['text'], str) or not row['text'].strip() or row['prompt'] != student_prompt(row):
            raise ValueError(f'Expected the current pointwise chat prompt matching the reward input: {row["id"]}')
    print(f'GRPO schema OK: {len(data)} conversational prompts')
    evaluation = None
    excluded_ids = []
    if args.eval_data:
        from datasets import Dataset
        heldout = [json.loads(line) for line in args.eval_data.read_text().splitlines() if line.strip()]
        if not heldout or len({row['id'] for row in heldout}) != len(heldout):
            raise ValueError('Evaluation data needs nonempty, unique record IDs')
        rows = list(data)
        indices = training_indices(rows, heldout)
        retained = {rows[i]['id'] for i in indices}
        excluded_ids = [row['id'] for row in rows if row['id'] not in retained]
        data = data.select(indices)
        evaluation = Dataset.from_list([dict(row, prompt=student_prompt(row)) for row in heldout])
        print(f'Excluded {len(excluded_ids)} training records related to {len(heldout)} development clauses')
    if args.check_data:
        return
    if not args.model.is_dir():
        parser.error('SFT checkpoint missing; run train_sft.py first')

    import torch
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from verifiqa.generation.llm_client import CodexCliClient
    from rl_rewards import make_grpo_reward
    if not torch.cuda.is_available():
        raise RuntimeError('A working CUDA GPU is required')
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), padding_side='left')
    data = data.filter(lambda row: len(tokenizer.apply_chat_template(
        row['prompt'], tokenize=True, return_dict=False, add_generation_prompt=True)) <= 3072)
    if not len(data):
        raise ValueError('No complete prompts fit the context budget')
    data = data.shuffle(seed=42)
    if args.train_limit:
        data = data.select(range(min(args.train_limit, len(data))))
    if evaluation is not None:
        for row in evaluation:
            if len(tokenizer.apply_chat_template(row['prompt'], tokenize=True, return_dict=False,
                                                 add_generation_prompt=True)) > 3072:
                raise ValueError(f'Evaluation prompt exceeds the context budget: {row["id"]}')
    print(f'GRPO uses {len(data)} complete training prompts')
    judge = CodexCliClient(model=os.environ.get('JUDGE_MODEL', 'gpt-5.6-luna'), reasoning_effort='medium', timeout=240)
    from autoformalization import JUDGE
    judge_prompt = JUDGE
    if args.parameter_comments:
        from parameter_semantics import with_parameter_semantics
        judge_prompt = with_parameter_semantics(JUDGE)
    if args.dense_reward:
        from rl_rewards import DENSE_RUBRIC
        judge_prompt += '\n' + DENSE_RUBRIC
    if args.non_temporal_reward:
        from probe_rewards import NON_TEMPORAL_SCOPE
        judge_prompt += '\n' + NON_TEMPORAL_SCOPE
    reward = make_grpo_reward(judge, judge_prompt=judge_prompt, dense=args.dense_reward)
    if args.validity_only_reward:
        from autoformalization import validate_smt
        text_of = lambda c: c[0]['content'] if isinstance(c, list) else c
        reward = lambda completions, **_: [float(validate_smt(text_of(c))['valid']) for c in completions]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mutation_reward:
        from mutation_rewards import make_mutation_grpo_reward, REWARD_CONFIG
        mutation_log = args.output_dir / 'mutation_rewards.jsonl'
        mutation_log.write_text('')
        def log_mutation(result):
            with mutation_log.open('a') as handle:
                handle.write(json.dumps(dict(result, phase='train' if trainer.model.training else 'eval')) + '\n')
        reward = make_mutation_grpo_reward(reward, log_mutation)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = str(args.model)
    adapter = (args.model / 'adapter_config.json').is_file()
    if adapter:
        from lora_utils import load_adapter, verify_adapter
        model = load_adapter(args.model, dtype=dtype, trainable=True)
        eos_ids = model.generation_config.eos_token_id
    else:
        from transformers import GenerationConfig
        eos_ids = GenerationConfig.from_pretrained(str(args.model)).eos_token_id
    endings = set(eos_ids if isinstance(eos_ids, list) else [eos_ids]) | {tokenizer.pad_token_id}
    logged_reward = make_logged_reward(reward, args.output_dir, num_generations=args.num_generations,
        phase_getter=lambda: 'train' if trainer.model.training else 'eval', ending_token_ids=endings,
        abstain_on_missing_context=args.dense_reward)

    config = GRPOConfig(
        output_dir=str(args.output_dir), max_steps=args.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        per_device_eval_batch_size=args.num_generations,
        max_completion_length=args.max_completion_length, learning_rate=args.learning_rate,
        loss_type='grpo', scale_rewards='group', beta=0.0,
        temperature=0.8, top_p=0.95,  # T=1.0 gave 18% valid SMT vs 40% at 0.8 (sampling sweep)
        remove_unused_columns=False, use_vllm=False, mask_truncated_completions=True,
        model_init_kwargs=None if adapter else {'dtype': dtype}, bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
        logging_steps=1, save_strategy='steps' if evaluation is not None else 'no',
        # Transformers also evaluates at the final step. Short diagnostics use the
        # separate, matched greedy evaluation instead of an extra sampled pass.
        eval_strategy='steps' if (evaluation is not None and not args.skip_trainer_eval
                                  and args.checkpoint_steps <= args.max_steps) else 'no',
        save_steps=args.checkpoint_steps, eval_steps=args.checkpoint_steps, save_total_limit=None,
        report_to='none', seed=42,
    )
    if args.dense_reward and config.generation_batch_size != args.num_generations:
        raise ValueError('Dense pilot requires one complete prompt group per generation batch')
    parent_config = args.model.parent / 'run_config.json'
    metadata = {'sft_checkpoint': str(args.model.resolve()),
                'sft_run': json.loads(parent_config.read_text()) if parent_config.exists() else None,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'excluded_development_record_ids': excluded_ids,
                'evaluation_sha256': hashlib.sha256(args.eval_data.read_bytes()).hexdigest() if args.eval_data else None,
                'evaluation_record_ids': list(evaluation['id']) if evaluation is not None else [],
                'record_ids': list(data['id']), 'judge': judge.run_metadata, 'training': config.to_dict(),
                'versions': {name: version(name) for name in
                             ('torch', 'trl', 'transformers', 'tokenizers', 'datasets', 'accelerate', 'huggingface-hub', 'z3-solver')}}
    metadata['reward_mode'] = 'validity_only' if args.validity_only_reward else 'dense' if args.dense_reward else 'mutation' if args.mutation_reward else 'base'
    if args.dense_reward:
        metadata['semantic_scope'] = 'non_temporal' if args.non_temporal_reward else 'full'
        metadata['context_policy'] = 'Return None for the whole prompt group; TRL excludes it from reward normalization and sets its advantages to zero.'
    metadata['judge_prompt_sha256'] = hashlib.sha256(judge_prompt.encode()).hexdigest()
    metadata['parameter_comments'] = args.parameter_comments
    metadata['stop_if_no_signal_after_groups'] = args.stop_if_no_signal_after
    if adapter:
        metadata['versions']['peft'] = version('peft')
    if args.mutation_reward:
        metadata['mutation_reward'] = REWARD_CONFIG
    if args.baseline_config:
        assert_matches_baseline(metadata, json.loads(args.baseline_config.read_text()))
        metadata['baseline_config_sha256'] = hashlib.sha256(args.baseline_config.read_bytes()).hexdigest()
        metadata['baseline_settings_verified'] = True
        print('Verified identical baseline student, data, judge, versions, and training settings')
    trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                          reward_funcs=logged_reward, train_dataset=data, eval_dataset=evaluation, args=config)
    if adapter:
        metadata['adapter_continuity'] = verify_adapter(trainer.model, args.model)
        metadata['trainable_parameters'] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print('Verified GRPO starts from the saved SFT adapter weights', flush=True)
    if args.stop_if_no_signal_after:
        class StopWithoutSignal(TrainerCallback):
            def on_step_end(self, training_args, state, control, **kwargs):
                groups = [json.loads(line) for line in (args.output_dir / 'reward_groups.jsonl').read_text().splitlines()]
                if no_reward_signal(groups, args.stop_if_no_signal_after):
                    control.should_training_stop = True
                    reason = {'step': state.global_step, 'reason': 'No effective informative training reward groups',
                              'minimum_groups': args.stop_if_no_signal_after}
                    (args.output_dir / 'early_stop.json').write_text(json.dumps(reason, indent=2) + '\n')
                    print(reason['reason'] + '; stopping for diagnosis', flush=True)
                return control
        trainer.add_callback(StopWithoutSignal())
    (args.output_dir / 'run_config.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    trainer.train()
    trainer.save_model(str(args.output_dir / 'final'))
    tokenizer.save_pretrained(str(args.output_dir / 'final'))
    trainer.save_state()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""exec_rl trainer: GRPO with the same settings as run_main, only the start checkpoint and the reward differ (PLAN.md).

Rewards:
  validity  1 if the SMT is valid in the supported fragment, else 0                          (run A)
  exec      0 if invalid; else 0.3 + 0.7 x execution score (exec_reward.score), with the       (runs B, C)
            status facts from a FIXED binder model; clauses without an extractable limit get 1.0 for the execution part

Reuses train_grpo.py helpers unchanged (exclusions, reward logging, no-signal stop). pool_ratio.jsonl holds run_main's 1,000
clauses in run_main's dataset order, and TRL's seeded sampler (seed 42, as in run_main) then draws them in the same order.

    python src/training/train_exec_grpo.py --model models/sft789/final --data data/rl/pool_fresh.jsonl \
        --reward exec_sig --max-steps 1000 --output-dir models/rl_run
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

from train_grpo import training_indices, make_logged_reward, no_reward_signal  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', type=Path, required=True, help='SFT adapter to start from')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--reward', choices=['validity', 'exec', 'exec_sig', 'exec_bin', 'exec_sched'], required=True,
                   help='exec_sig: headroom RL v2 (docs/headroom_PLAN.md) - 0 unless the formula is valid and keeps exactly '
                        'the required parameters (row "params"), else 0.3 + 0.7 x execution score with test_date fixed late. '
                        'exec_bin: ablation of exec_sig - same gate and tests, but 1 only if every test passes, else 0. '
                        'exec_sched: exec_sig run over every schedule window and condition setting of the clause '
                        '(row "spec_json", data/rl/rl_v3); identical to exec_sig on clauses without either')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--learning-rate', type=float, default=5e-6)
    p.add_argument('--num-generations', type=int, default=4)
    p.add_argument('--max-completion-length', type=int, default=1024)
    p.add_argument('--eval-data', type=Path, default=P.VALIDATION_250,
                   help='held-out clauses; related training records are excluded (as in run_main)')
    p.add_argument('--checkpoint-steps', type=int, default=250)
    p.add_argument('--stop-if-no-signal-after', type=int, default=50)
    p.add_argument('--binder-model', default='Qwen/Qwen2.5-Coder-7B-Instruct')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from autoformalization import validate_smt
    from parameter_semantics import student_prompt
    from lora_utils import load_adapter, verify_adapter

    data = load_dataset('json', data_files=str(args.data), split='train')
    for row in data:
        if args.reward not in ('exec_sig', 'exec_bin', 'exec_sched') and row['prompt'] != student_prompt(row):
            raise ValueError(f'prompt does not match the parameter-comment student prompt: {row["id"]}')
    heldout = [json.loads(l) for l in args.eval_data.read_text().splitlines() if l.strip()]
    rows = list(data)
    indices = training_indices(rows, heldout)
    excluded = [r['id'] for i, r in enumerate(rows) if i not in set(indices)]
    data = data.select(indices)
    from lora_utils import load_tokenizer  # DeepSeek-Coder: tokenizer that round-trips spaces and newlines
    tokenizer = load_tokenizer(str(args.model), padding_side='left')
    data = data.filter(lambda row: len(tokenizer.apply_chat_template(
        row['prompt'], tokenize=True, return_dict=False, add_generation_prompt=True)) <= 3072)
    print(f'{len(data)} training prompts ({len(excluded)} excluded as related to held-out clauses)', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    text_of = lambda c: c[0]['content'] if isinstance(c, list) else c
    if args.reward in ('exec_sig', 'exec_bin', 'exec_sched'):
        import re
        import z3
        from exec_reward import score, score_spec
        exec_log = args.output_dir / 'exec_rewards.jsonl'
        args.output_dir.mkdir(parents=True, exist_ok=True)
        exec_log.write_text('')

        def smt_of(c):
            t = re.sub(r'```(?:smt2?|smtlib2?|lisp|scheme)?', '', text_of(c))
            m = re.search(r'\(\s*define-fun\s+covenant_holds', t)
            sl = re.search(r'\(\s*set-logic[^)]*\)', t)
            return ((sl.group(0) + '\n') if sl else '') + (t[m.start():].strip() if m else t.strip())

        def sig_ok(smt, params):
            m = re.search(r'define-fun\s+covenant_holds\s*\(((?:\s*\(\s*[^()\s]+\s+[^()\s]+\s*\))*)\s*\)\s*Bool', smt)
            got = re.findall(r'\(\s*([^()\s]+)\s+([^()\s]+)\s*\)', m.group(1)) if m else None
            if got != [tuple(x) for x in params]:
                return False
            try:  # parses with every parameter declared
                decls = ''.join(f'(declare-const {n} {t})' for n, t in params)
                z3.parse_smt2_string(re.sub(r'\(\s*set-logic[^)]*\)', '', smt) + decls +
                                     f"(assert (covenant_holds {' '.join(n for n, _ in params)}))")
                return True
            except z3.Z3Exception:
                return False

        def reward(completions, text, **kwargs):
            out = []
            with exec_log.open('a') as handle:
                for i, (c, t) in enumerate(zip(completions, text)):
                    smt = smt_of(c)
                    if not sig_ok(smt, kwargs['params'][i]):
                        out.append(0.0)
                        handle.write(json.dumps({'id': kwargs['id'][i], 'valid_signature': False, 'reward': 0.0}) + '\n')
                        continue
                    r = (score_spec(t, smt, json.loads(kwargs['spec_json'][i])) if args.reward == 'exec_sched'
                         else score(t, smt, '', [z3.Int('test_date') == 20991231]))
                    x = 1.0 if r['score'] is None else r['score']
                    out.append(float(x == 1.0) if args.reward == 'exec_bin' else 0.3 + 0.7 * x)
                    handle.write(json.dumps({'id': kwargs['id'][i], 'valid_signature': True, 'exec': r['score'],
                                             'anchors': r.get('anchors'), 'matched': r.get('matched'), 'reward': out[-1]}) + '\n')
            return out
        binder = None
    elif args.reward == 'validity':
        def reward(completions, **_):
            return [float(validate_smt(text_of(c))['valid']) for c in completions]
        binder = None
    else:
        from exec_reward import Binder, score
        binder = Binder(args.binder_model)
        exec_log = args.output_dir / 'exec_rewards.jsonl'
        exec_log.write_text('')

        def reward(completions, text, **kwargs):
            smts = [text_of(c) for c in completions]
            valid = [validate_smt(s)['valid'] for s in smts]
            facts = dict(zip([i for i, v in enumerate(valid) if v], binder.bind([s for s, v in zip(smts, valid) if v])
                             if any(valid) else []))
            out = []
            with exec_log.open('a') as handle:
                for i, (s, t, v) in enumerate(zip(smts, text, valid)):
                    if not v:
                        out.append(0.0)
                        handle.write(json.dumps({'id': kwargs['id'][i], 'valid': False, 'reward': 0.0}) + '\n')
                        continue
                    r = score(t, s, facts[i])
                    x = 1.0 if r['score'] is None else r['score']
                    out.append(0.3 + 0.7 * x)
                    handle.write(json.dumps({'id': kwargs['id'][i], 'valid': True, 'exec': r['score'], 'anchors': r['anchors'],
                                             'matched': r['matched'], 'passed': r['passed'], 'scenarios': r['scenarios'],
                                             'facts': facts[i], 'details': r['details'][:6], 'reward': out[-1]}) + '\n')
            return out

    dtype = torch.bfloat16
    model = load_adapter(args.model, dtype=dtype, trainable=True)
    eos = model.generation_config.eos_token_id
    endings = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.pad_token_id}
    trainer = None
    logged = make_logged_reward(reward, args.output_dir, num_generations=args.num_generations,
                                phase_getter=lambda: 'train' if trainer.model.training else 'eval', ending_token_ids=endings)
    config = GRPOConfig(
        output_dir=str(args.output_dir), max_steps=args.max_steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.num_generations, num_generations=args.num_generations,
        per_device_eval_batch_size=args.num_generations, max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate, loss_type='grpo', scale_rewards='group', beta=0.0, temperature=0.8, top_p=0.95,
        remove_unused_columns=False, use_vllm=False, mask_truncated_completions=True, bf16=True, fp16=False,
        gradient_checkpointing=True, logging_steps=1, save_strategy='steps', eval_strategy='no',
        save_steps=args.checkpoint_steps, save_total_limit=None, report_to='none', seed=args.seed)  # TRL's default seeded shuffle, as in run_main
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=logged, train_dataset=data, args=config)

    metadata = {'experiment': 'exec_rl', 'start_checkpoint': str(args.model.resolve()), 'reward': args.reward,
                'binder_model': binder.name if binder else None, 'data': str(args.data),
                'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(), 'record_ids': list(data['id']),
                'excluded_related_record_ids': excluded, 'eval_data': str(args.eval_data), 'seed': args.seed,
                'adapter_continuity': verify_adapter(trainer.model, args.model), 'training': config.to_dict(),
                'versions': {n: version(n) for n in ('torch', 'trl', 'transformers', 'peft', 'z3-solver')}}
    (args.output_dir / 'run_config.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')

    class StopWithoutSignal(TrainerCallback):
        def on_step_end(self, training_args, state, control, **kwargs):
            groups = [json.loads(l) for l in (args.output_dir / 'reward_groups.jsonl').read_text().splitlines()]
            if no_reward_signal(groups, args.stop_if_no_signal_after):
                control.should_training_stop = True
                (args.output_dir / 'early_stop.json').write_text(json.dumps({'step': state.global_step}) + '\n')
            return control
    if args.stop_if_no_signal_after:
        trainer.add_callback(StopWithoutSignal())
    trainer.train()
    trainer.save_model(str(args.output_dir / 'final'))
    tokenizer.save_pretrained(str(args.output_dir / 'final'))
    trainer.save_state()
    print('EXEC_GRPO_DONE', flush=True)


if __name__ == '__main__':
    main()

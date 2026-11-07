#!/usr/bin/env python3
"""TRL SFT on accepted real-covenant NL-SMT pairs; defaults to a five-step smoke test."""
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

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--revision', default='main')
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument('--max-steps', type=int, default=5)
    duration.add_argument('--num-train-epochs', type=float)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--lora-rank', type=int, default=0, help='0 keeps full fine-tuning')
    parser.add_argument('--load-in-4bit', action='store_true', help='QLoRA: load the base model in 4-bit NF4 (bf16 compute)')
    parser.add_argument('--data', type=Path, default=P.SFT_PAIRS)
    parser.add_argument('--output-dir', type=Path, default=P.MODELS / 'sft_run')
    parser.add_argument('--check-data', action='store_true', help='Check the JSONL schema and exit without loading a model')
    args = parser.parse_args()
    if (args.max_steps < 1 or args.learning_rate <= 0 or args.lora_rank < 0
            or (args.num_train_epochs is not None and args.num_train_epochs <= 0)):
        parser.error('Training duration and learning rate must be positive; LoRA rank must be nonnegative')

    from datasets import load_dataset
    source = args.data
    data = load_dataset('json', data_files=str(source), split='train')
    if not len(data) or 'messages' not in data.column_names:
        raise ValueError('SFT data must contain nonempty messages records')
    for row in data:
        messages = row['messages']
        if (not isinstance(messages, list) or len(messages) < 2
                or any(not isinstance(m, dict) or m.get('role') not in ('system', 'user', 'assistant')
                       or not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages)
                or messages[-1]['role'] != 'assistant' or messages[-2]['role'] != 'user'):
            raise ValueError(f'Expected chat messages ending in a user/assistant pair: {row["id"]}')
    print(f'SFT schema OK: {len(data)} pairs')
    if args.check_data:
        return

    import torch
    from huggingface_hub import model_info
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available():
        raise RuntimeError('A working CUDA GPU is required for this smoke test')
    revision = model_info(args.model, revision=args.revision).sha
    from lora_utils import load_tokenizer
    tokenizer = load_tokenizer(args.model, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    total = len(data)
    data = data.filter(lambda row: len(tokenizer.apply_chat_template(
        row['messages'], tokenize=True, return_dict=False)) <= 4096)
    print(f'Complete examples fitting 4096 tokens: {len(data)}/{total}')
    if not len(data):
        raise ValueError('No complete examples fit the context budget')
    record_ids, sample = data['id'], data[0]
    data = data.map(lambda row: {'prompt': row['messages'][:-1], 'completion': row['messages'][-1:]},
                    remove_columns=data.column_names)
    bf16 = torch.cuda.is_bf16_supported()
    # Full FP16 weights conflict with AMP gradient scaling; keep FP32 masters on the fallback path.
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = SFTConfig(
        output_dir=str(args.output_dir), max_steps=-1 if args.num_train_epochs is not None else args.max_steps,
        num_train_epochs=args.num_train_epochs if args.num_train_epochs is not None else 3.0,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=args.learning_rate,
        max_length=4096, packing=False, completion_only_loss=True,
        eos_token=tokenizer.eos_token, model_init_kwargs=dict({'dtype': dtype, 'revision': revision}, **(
            {'quantization_config': __import__('transformers').BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
             bnb_4bit_compute_dtype=dtype)} if args.load_in_4bit else {})),
        bf16=bf16, fp16=not bf16, gradient_checkpointing=True,
        logging_steps=1, save_strategy='no', eval_strategy='no', report_to='none', seed=42,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {'model': args.model, 'model_revision': revision, 'tokenizer_revision': revision,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'record_ids': list(record_ids), 'training': config.to_dict(),
                'versions': {name: version(name) for name in
                             ('torch', 'trl', 'transformers', 'tokenizers', 'datasets', 'accelerate', 'huggingface-hub')}}
    peft_config = None
    if args.lora_rank:
        from peft import LoraConfig
        peft_config = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
            lora_dropout=0.0, target_modules='all-linear', bias='none', task_type='CAUSAL_LM', revision=revision)
        metadata['lora'] = peft_config.to_dict()
        metadata['versions']['peft'] = version('peft')
    trainer = SFTTrainer(model=args.model, processing_class=tokenizer, train_dataset=data,
                         args=config, peft_config=peft_config)
    if peft_config is not None:
        trainable = [(name, param) for name, param in trainer.model.named_parameters() if param.requires_grad]
        assert trainable and all('lora_' in name for name, _ in trainable)
        metadata['trainable_parameters'] = sum(param.numel() for _, param in trainable)
        metadata['trainable_dtypes'] = sorted({str(param.dtype) for _, param in trainable})
        trainer.model.print_trainable_parameters()
    (args.output_dir / 'run_config.json').write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    trainer.train()
    trainer.save_model(str(args.output_dir / 'final'))
    tokenizer.save_pretrained(str(args.output_dir / 'final'))
    trainer.save_state()

    # A training-input sample for inspecting output format, not a held-out evaluation.
    trainer.model.eval()
    inputs = tokenizer.apply_chat_template(sample['messages'][:-1], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt').to(trainer.model.device)
    with torch.no_grad():
        generated = trainer.model.generate(**inputs, max_new_tokens=512, do_sample=False)
    output = tokenizer.decode(generated[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    (args.output_dir / 'sample.json').write_text(json.dumps(
        {'id': sample['id'], 'split': 'train', 'purpose': 'format_smoke_test', 'smt2': output}, indent=2) + '\n')


if __name__ == '__main__':
    main()

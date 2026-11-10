"""Batched greedy generation with a local Hugging Face model or LoRA adapter (shared by every generation step).

load(model) -> (tokenizer, model); run(...) writes one JSON line per item with the decoded output, resume-safe.
Extracted verbatim from the downstream fact-writing pipeline.
"""
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SRC)] + [str(d) for d in sorted(SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))]
from paths import ROOT  # noqa: E402

NO_THINKING = {'Qwen/Qwen3.8-27B'}
FOUR_BIT = {'meta-llama/Llama-3.3-70B-Instruct'}  # 70B reader: bf16 needs ~140 GB > 96 GB GPU; bitsandbytes NF4 (~40 GB). FP8 needs Hub kernels; INT8 is ~11x slower


def load(writer):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = ROOT / writer if (ROOT / writer).exists() else Path(writer)
    from lora_utils import load_tokenizer  # DeepSeek-Coder: tokenizer that round-trips spaces and newlines
    tok = load_tokenizer(str(path), padding_side='left')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16
    if (path / 'adapter_config.json').is_file():
        from lora_utils import load_adapter
        model = load_adapter(path, dtype=dtype)
    elif writer in FOUR_BIT:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(str(path), device_map={'': 0}, dtype=dtype,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype))
        return tok, model.eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    return tok, model.to('cuda').eval()


def run(tok, model, writer, items, out_path, batch_size, max_new_tokens, max_prompt_tokens):
    import torch
    kwargs = {'enable_thinking': False} if writer in NO_THINKING else {}
    done = set()
    if out_path.exists():
        done = {json.loads(l)['case_id'] for l in open(out_path)}
    todo = [it for it in items if it['case_id'] not in done]
    lengths = {it['case_id']: len(tok.apply_chat_template(it['messages'], tokenize=True, return_dict=False,
                                                          add_generation_prompt=True, **kwargs)) for it in todo}
    todo.sort(key=lambda it: lengths[it['case_id']])
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    t0 = time.time()
    with open(out_path, 'a') as handle:
        too_long = [it for it in todo if lengths[it['case_id']] > max_prompt_tokens]
        for it in too_long:
            handle.write(json.dumps(dict(case_id=it['case_id'], clause_id=it['clause_id'], writer=writer,
                                         output='', error='prompt_too_long', prompt_tokens=lengths[it['case_id']])) + '\n')
        todo = [it for it in todo if lengths[it['case_id']] <= max_prompt_tokens]
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            inputs = tok.apply_chat_template([it['messages'] for it in batch], tokenize=True, add_generation_prompt=True,
                                             padding=True, return_dict=True, return_tensors='pt', **kwargs).to('cuda')
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
            for it, seq in zip(batch, gen):
                toks = seq[inputs['input_ids'].shape[-1]:].tolist()
                end = next((i for i, t in enumerate(toks) if t in eos), None)
                hit = end is None and len(toks) >= max_new_tokens
                text = tok.decode(toks[:end] if end is not None else toks, skip_special_tokens=True)
                handle.write(json.dumps(dict(case_id=it['case_id'], clause_id=it['clause_id'], writer=writer, output=text,
                                             error='hit_token_limit' if hit else '', prompt_tokens=lengths[it['case_id']])) + '\n')
            handle.flush()
            n = start + len(batch)
            print(f'  {out_path.parent.name}/{out_path.stem}: {n}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)', flush=True)



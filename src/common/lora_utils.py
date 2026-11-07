"""Load a saved adapter on its pinned base and verify its weights before continuing."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import hashlib
from pathlib import Path


def verify_adapter(model, directory):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    path = Path(directory) / 'adapter_model.safetensors'
    saved = load_file(str(path))
    loaded = get_peft_model_state_dict(model)
    if set(saved) != set(loaded) or any(
        not torch.equal(value.cpu(), loaded[name].detach().cpu().to(value.dtype))
        for name, value in saved.items()
    ):
        raise ValueError('Loaded adapter does not exactly match the saved SFT weights')
    return {'adapter_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'matching_tensors': len(saved), 'weights_match_saved_adapter': True}


def load_adapter(directory, *, dtype, trainable=False):
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM
    config = PeftConfig.from_pretrained(str(directory))
    if not config.revision:
        raise ValueError('Adapter must record the pinned base-model revision')
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                             revision=config.revision, dtype=dtype)
    model = PeftModel.from_pretrained(base, str(directory), is_trainable=trainable)
    verify_adapter(model, directory)
    if trainable:
        names = [name for name, param in model.named_parameters() if param.requires_grad]
        if not names or any('lora_' not in name for name in names):
            raise ValueError('Expected only the saved LoRA adapter to be trainable')
    return model


def load_tokenizer(name, **kwargs):
    """AutoTokenizer, falling back to the plain fast tokenizer when decoding does not round-trip.

    DeepSeek-Coder's byte-level tokenizer.json loads under transformers 5 as a Llama tokenizer that drops
    spaces and newlines; the plain fast tokenizer reads tokenizer.json faithfully.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    probe = '(set-logic QF_LRA)\n(define-fun f ((x Real)) Bool (<= x 3.5))'
    if tokenizer.decode(tokenizer.encode(probe, add_special_tokens=False)) != probe:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(name, **kwargs)
        assert tokenizer.decode(tokenizer.encode(probe, add_special_tokens=False)) == probe
    return tokenizer

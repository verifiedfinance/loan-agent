"""A saved, nonzero adapter must survive loading for GRPO unchanged."""
from pathlib import Path
import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from lora_utils import load_adapter, verify_adapter
from train_grpo import no_reward_signal


class LoRAContinuationTests(unittest.TestCase):
    def test_saved_adapter_is_loaded_and_only_adapter_is_trainable(self):
        with tempfile.TemporaryDirectory() as directory:
            base_path, adapter_path = Path(directory) / 'base', Path(directory) / 'adapter'
            tiny = Qwen2ForCausalLM(Qwen2Config(vocab_size=32, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2))
            tiny.save_pretrained(base_path)
            base = AutoModelForCausalLM.from_pretrained(base_path)
            model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4,
                target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', revision='test-pinned-revision'))
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'lora_B' in name:
                        param.fill_(0.125)
            model.save_pretrained(adapter_path)
            loaded = load_adapter(adapter_path, dtype=torch.float32, trainable=True)
            self.assertTrue(verify_adapter(loaded, adapter_path)['weights_match_saved_adapter'])
            trainable = [(name, param) for name, param in loaded.named_parameters() if param.requires_grad]
            self.assertTrue(trainable)
            self.assertTrue(all('lora_' in name for name, _ in trainable))
            with torch.no_grad():
                next(param for name, param in trainable if 'lora_B' in name).add_(0.01)
            with self.assertRaisesRegex(ValueError, 'does not exactly match'):
                verify_adapter(loaded, adapter_path)

    def test_stop_requires_enough_known_training_groups_without_usable_signal(self):
        empty = dict(phase='train', complete_groups=1, effective_informative_groups=0)
        self.assertFalse(no_reward_signal([empty] * 4, 5))
        self.assertTrue(no_reward_signal([empty] * 5, 5))
        self.assertFalse(no_reward_signal([empty] * 4 + [dict(empty, effective_informative_groups=1)], 5))
        self.assertFalse(no_reward_signal([dict(empty, phase='eval')] * 5, 5))
        self.assertFalse(no_reward_signal([dict(empty, effective_informative_groups=None)] * 5, 5))


if __name__ == '__main__':
    unittest.main()

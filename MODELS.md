# Models

| Role | Hugging Face id | Revision |
|---|---|---|
| Base of every trained adapter; prompted 7B formalizer; 7B direct reader | Qwen/Qwen2.5-Coder-7B-Instruct | c03e6d358207e414f1eca0bb1891e29f1db0e242 |
| Reference reader B (benchmark construction); direct reader | Qwen/Qwen3.8-27B | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 |
| Direct reader | Qwen/Qwen2.5-Coder-32B-Instruct | from the shared HF cache |

API model (through the `claude` command-line tool): `claude-sonnet-5-5` is the SFT teacher and judge, reference
reader A, and the RL v2 item namer.

Trained adapters, expected in `models/` (not included in this repository; LoRA rank 16 on Qwen2.5-Coder-7B-Instruct;
`final/` is the adapter, `run_config.json` the exact training configuration):

| Folder | What |
|---|---|
| `sft789/` | SFT on the 789 teacher pairs (`data/sft/sft_pairs.jsonl`) |
| `rl_v1/` | SFT + GRPO, execution reward, old prompt format (reported as a failure: breaks the formula interface) |
| `rl_v2_s42/`, `rl_v2_s43/`, `rl_v2_s44/` | SFT + GRPO, interface-enforcing execution reward (`exec_sig`), three seeds; the paper's model |

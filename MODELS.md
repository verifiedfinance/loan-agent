# Models

| Role | Hugging Face id | Revision |
|---|---|---|
| Base of the Qwen adapters; prompted 7B formalizer; 7B direct reader | Qwen/Qwen2.5-Coder-7B-Instruct | c03e6d358207e414f1eca0bb1891e29f1db0e242 |
| Base of the Llama adapters; prompted formalizer; direct reader | meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |
| Base of the Mistral adapters; prompted formalizer; direct reader | mistralai/Mistral-7B-Instruct-v0.3 | c170c708c41dac9275d15a8fff4eca08d52bab71 |
| Reference reader B (benchmark construction); direct reader | Qwen/Qwen3.8-27B | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 |
| Direct reader | Qwen/Qwen2.5-Coder-32B-Instruct | from the shared HF cache |
| Direct reader, loaded in 4-bit NF4 (bitsandbytes) | meta-llama/Llama-3.3-70B-Instruct | 6f6073b423013f6a7d4d9f39144961bfbfbc386b |
| Set up only, not evaluated | deepseek-ai/deepseek-coder-6.7b-instruct | e5d64addd26a6a1db0f9b863abf6ee3141936807 |

API model (through the `claude` command-line tool): `claude-sonnet-5-5` is the SFT teacher and judge, reference
reader A, the RL item namer, and one of the LLM analyses checked by FormaFin.

Trained adapters, expected in `models/` (not included in this repository; LoRA rank 16, scaling 32, all linear layers;
`final/` is the adapter, `checkpoint-250/500/750/1000/` the GRPO checkpoints, `run_config.json` the exact training
configuration, `exec_rewards.jsonl` every sampled formula's reward):

| Folder | What |
|---|---|
| `sft789/` | Qwen2.5-Coder-7B, SFT on the 789 teacher pairs (`data/sft/sft_pairs.jsonl`) |
| `rl_sched_s42/`, `rl_sched_s43/`, `rl_sched_s44/` | **FormaFin (final model)**: SFT + GRPO with the final reward (`exec_sched`, `data/rl/rl_v3`), three seeds |
| `rl_v2_s42/`, `rl_v2_s43/`, `rl_v2_s44/` | SFT + GRPO with the first, single-date reward (`exec_sig`, `data/rl/rl_v2`) |
| `rl_bin_s42/`, `rl_bin_s43/`, `rl_bin_s44/` | ablation: pass/fail reward (`exec_bin`) |
| `rl_v1/` | SFT + GRPO without the interface gate, old prompt format (reported as a failure) |
| `llama_sft789/`, `llama_rl_sched_s42/43/44/` | Llama-3.1-8B: SFT, then GRPO with the final reward |
| `llama_rl_v2_s42/43/44/` | Llama-3.1-8B with the first reward (seed 42 learned the invented date guard) |
| `mistral_sft789/`, `mistral_rl_v2_s42/43/44/` | Mistral-7B: SFT, then GRPO with the final reward (`exec_sched`, despite the folder name) |

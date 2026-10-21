# GRPO on GSM8K: a reproducible baseline for Qwen2.5-1.5B

Group Relative Policy Optimization (GRPO) with a single, verifiable reward — exact match on the
final numeric answer — applied to `Qwen/Qwen2.5-1.5B` on GSM8K, trained on one Colab A100.
The repo is the whole loop: data → base eval → RL training → trained eval → statistics → this
README, all driven from one config and runnable with one command.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/UbajakaCJ/grpo-gsm8k/blob/main/notebooks/colab_a100.ipynb)
![ci](https://github.com/UbajakaCJ/grpo-gsm8k/actions/workflows/ci.yml/badge.svg)

## Results

<!-- RESULTS:START -->
_Not run yet. `make all` fills this section from `results/` automatically._

> Ran a GRPO baseline on Qwen2.5-1.5B with exact-match math rewards: GSM8K accuracy went from [X]% to [Y]% after [N] training steps on one Colab A100. Code: github.com/UbajakaCJ/grpo-gsm8k
<!-- RESULTS:END -->

## Quickstart

**Colab (recommended):** open the badge above, pick an A100 runtime, *Run all*. The notebook runs
the unit tests, a 5-minute smoke run, the full experiment, and pushes `results/` back here.

**Any CUDA box:**

```bash
git clone https://github.com/UbajakaCJ/grpo-gsm8k && cd grpo-gsm8k
pip install vllm && pip install -e ".[train,dev]"
make test          # CPU unit tests (what CI runs)
make smoke         # whole pipeline, tiny model, 4 steps
make all           # eval-base -> train -> eval-trained -> report
```

Override any config key from the command line:

```bash
python -m grpo_gsm8k.train --config configs/colab_a100.yaml train.max_steps=300 train.report_to=wandb
```

## What the experiment is

| | |
|---|---|
| Policy | `Qwen/Qwen2.5-1.5B` (base), full fine-tune, bf16, gradient checkpointing |
| Data | GSM8K `main` train split (7,473 problems); evaluated on the full test split (1,319) |
| Prompt | Chat template + system prompt asking for step-by-step reasoning and a final `#### <answer>` line |
| Reward | `1.0` if the strictly extracted final answer equals the gold answer, else `0.0` |
| Algorithm | TRL `GRPOTrainer`, 8 generations per prompt, 8 prompts per step, `dr_grpo` loss, β = 0 (no KL / reference model) |
| Rollouts | vLLM colocated on the same GPU |
| Eval | Zero-shot, greedy, same prompt for base and trained model, 512 new tokens |

**Strict vs flexible accuracy.** The headline number and the reward both use *strict* extraction:
only an explicit `#### x` line or `\boxed{x}` counts, after normalising commas, `$`, `%`,
trailing zeros and simple fractions. *Flexible* accuracy (falls back to "the answer is x", then the
last number in the text) is reported next to it as a diagnostic. Part of any GRPO gain on a base
model is learning to state an answer in the expected format; reporting both, plus the
no-final-answer rate, keeps that visible rather than hidden in the headline.

**Significance.** The report computes a paired 95% bootstrap CI for the accuracy change and an
exact McNemar test over the per-problem outcomes, and counts problems gained and lost.

## Repository layout

```
configs/
  colab_a100.yaml        main experiment (all hyperparameters live here)
  smoke.yaml             inherits the main config; tiny model, 4 steps
src/grpo_gsm8k/
  rewards.py             answer extraction, normalisation, TRL reward functions
  data.py                GSM8K -> conversational prompts (shared by train and eval)
  train.py               GRPOTrainer entrypoint; saves merged model + train log/metadata
  evaluate.py            vLLM (or HF generate) evaluation; per-problem JSONL + summary
  report.py              headline sentence, stats, training curves, README update
  config.py, utils.py    config inheritance/overrides, seeding, run metadata
tests/                   reward, config, data and report tests (CPU only)
notebooks/colab_a100.ipynb   one-click Colab runner
.github/workflows/ci.yml     lint + tests on every push
```

## Artifacts and reproducibility

Every run writes to `results/`:

- `eval_base.json`, `eval_trained.json` — accuracy (strict and flexible), no-answer rate, decoding settings, package versions, GPU
- `eval_*.jsonl` — every test problem with completion, extracted prediction and correctness
- `train_meta.json` — steps, wall time, config hash, git commit, full resolved config
- `train_log.json` — TRL log history (reward, completion length, KL, clip ratio …)
- `summary.json`, `headline.md`, `training_curves.png`

Seeds are fixed and every artifact carries the config hash and git commit. Checkpoints are not
committed (see `.gitignore`); set `hub.push_to_hub: true` and `hub.repo_id` to push the trained
model to the Hugging Face Hub.

## Troubleshooting

- **OOM during training:** set `model.lora.enabled=true`, or lower `train.vllm_gpu_memory_utilization`, or `train.max_completion_length=384`.
- **vLLM install conflicts on Colab:** run with `train.use_vllm=false eval.use_vllm=false` (slower HF generation, same results).
- **TRL version drift:** unsupported config keys are dropped with a warning rather than crashing; the installed versions are recorded in every results file.

## License

MIT

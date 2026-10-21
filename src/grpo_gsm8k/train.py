"""GRPO training entrypoint.

    python -m grpo_gsm8k.train --config configs/colab_a100.yaml
    python -m grpo_gsm8k.train --config configs/colab_a100.yaml train.max_steps=200

Writes the final (merged) model to ``<output_dir>/final`` and the training log and
run metadata to ``<results_dir>/train_log.json`` and ``train_meta.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path

from .config import config_hash, load_config
from .data import load_train
from .rewards import correctness_reward, format_reward
from .utils import dtype_kwarg, env_info, set_seed, write_json

log = logging.getLogger("grpo_gsm8k.train")


def build_grpo_config(cfg: dict):
    """Translate our YAML into a ``trl.GRPOConfig``, tolerating TRL version drift."""
    from trl import GRPOConfig

    t = cfg["train"]
    r = cfg["reward"]
    weights = [r["correctness_weight"]] + ([r["format_weight"]] if r.get("format_weight") else [])
    wanted = {
        "output_dir": cfg["output_dir"],
        "run_name": cfg["run_name"],
        "seed": cfg["seed"],
        "max_steps": t["max_steps"],
        "learning_rate": t["learning_rate"],
        "lr_scheduler_type": t["lr_scheduler_type"],
        "warmup_ratio": t["warmup_ratio"],
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "gradient_accumulation_steps": t["gradient_accumulation_steps"],
        "num_generations": t["num_generations"],
        "max_prompt_length": t["max_prompt_length"],
        "max_completion_length": t["max_completion_length"],
        "temperature": t["temperature"],
        "beta": t["beta"],
        "epsilon": t.get("epsilon", 0.2),
        "loss_type": t.get("loss_type"),
        "max_grad_norm": t["max_grad_norm"],
        "logging_steps": t["logging_steps"],
        "save_steps": t["save_steps"],
        "save_total_limit": t.get("save_total_limit"),
        "bf16": cfg["model"]["dtype"] == "bfloat16",
        "gradient_checkpointing": cfg["model"].get("gradient_checkpointing", False),
        "use_vllm": t["use_vllm"],
        "vllm_mode": t.get("vllm_mode"),
        "vllm_gpu_memory_utilization": t.get("vllm_gpu_memory_utilization"),
        "reward_weights": weights,
        "report_to": t.get("report_to", "none"),
        "log_completions": True,
        "num_completions_to_print": 2,
    }
    known = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k, v in wanted.items() if k not in known and v is not None)
    if dropped:
        log.warning("Installed TRL does not support %s; ignoring them.", dropped)
    kwargs = {k: v for k, v in wanted.items() if k in known and v is not None}
    return GRPOConfig(**kwargs)


def load_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(m["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        m["name_or_path"],
        **dtype_kwarg(getattr(torch, m["dtype"])),
        attn_implementation=m.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    return model, tokenizer


def build_peft_config(cfg: dict):
    lora = cfg["model"].get("lora") or {}
    if not lora.get("enabled"):
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        task_type="CAUSAL_LM",
    )


def train(cfg: dict) -> dict:
    from trl import GRPOTrainer

    set_seed(cfg["seed"])
    results_dir = Path(cfg["results_dir"])
    final_dir = Path(cfg["output_dir"]) / "final"

    train_ds = load_train(cfg)
    model, tokenizer = load_model_and_tokenizer(cfg)
    reward_funcs = [correctness_reward]
    if cfg["reward"].get("format_weight"):
        reward_funcs.append(format_reward)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=build_grpo_config(cfg),
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg),
    )

    start = time.time()
    trainer.train()
    wall = time.time() - start

    # Merge LoRA (if any) so evaluation always loads a plain checkpoint.
    model_out = trainer.model
    if hasattr(model_out, "merge_and_unload"):
        model_out = model_out.merge_and_unload()
    model_out.config.use_cache = True
    model_out.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    t = cfg["train"]
    meta = {
        "run_name": cfg["run_name"],
        "config_path": cfg.get("_config_path"),
        "config_hash": config_hash(cfg),
        "base_model": cfg["model"]["name_or_path"],
        "final_model_dir": str(final_dir),
        "global_steps": int(trainer.state.global_step),
        "train_problems": len(train_ds),
        "prompts_per_step": t["per_device_train_batch_size"]
        * t["gradient_accumulation_steps"]
        // t["num_generations"],
        "num_generations": t["num_generations"],
        "train_wall_seconds": round(wall, 1),
        "env": env_info(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    write_json(results_dir / "train_meta.json", meta)
    write_json(results_dir / "train_log.json", trainer.state.log_history)

    hub = cfg.get("hub") or {}
    if hub.get("push_to_hub") and hub.get("repo_id"):
        model_out.push_to_hub(hub["repo_id"], private=True)
        tokenizer.push_to_hub(hub["repo_id"], private=True)
        log.info("Pushed model to https://huggingface.co/%s", hub["repo_id"])

    log.info("Training done: %d steps in %.0fs -> %s", meta["global_steps"], wall, final_dir)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.max_steps=50")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()

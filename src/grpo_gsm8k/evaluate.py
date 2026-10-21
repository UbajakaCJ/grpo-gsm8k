"""Zero-shot greedy GSM8K evaluation with the same prompt used in training.

    python -m grpo_gsm8k.evaluate --config configs/colab_a100.yaml --tag base
    python -m grpo_gsm8k.evaluate --config configs/colab_a100.yaml --tag trained \\
        --model outputs/qwen2.5-1.5b-grpo-gsm8k/final

Writes ``<results_dir>/eval_<tag>.json`` (summary) and ``eval_<tag>.jsonl`` (per problem).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from .config import config_hash, load_config
from .data import load_eval
from .rewards import extract_answer
from .utils import dtype_kwarg, env_info, set_seed, write_json

log = logging.getLogger("grpo_gsm8k.evaluate")


def _vllm_available() -> bool:
    try:
        import torch
        import vllm  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


def generate_vllm(model_path: str, prompts: list[str], cfg: dict) -> list[str]:
    from vllm import LLM, SamplingParams

    e = cfg["eval"]
    llm = LLM(
        model=model_path,
        dtype=cfg["model"]["dtype"],
        seed=cfg["seed"],
        gpu_memory_utilization=e["vllm_gpu_memory_utilization"],
        max_model_len=cfg["train"]["max_prompt_length"] + e["max_new_tokens"] + 256,
    )
    params = SamplingParams(temperature=e["temperature"], max_tokens=e["max_new_tokens"])
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def generate_hf(model_path: str, prompts: list[str], tokenizer, cfg: dict) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM

    e = cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg["model"]["dtype"]) if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, **dtype_kwarg(dtype)).to(device).eval()
    tokenizer.padding_side = "left"
    greedy = e["temperature"] == 0
    texts: list[str] = []
    for i in range(0, len(prompts), e["batch_size"]):
        batch = tokenizer(prompts[i : i + e["batch_size"]], return_tensors="pt", padding=True)
        batch = batch.to(device)
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=e["max_new_tokens"],
                do_sample=not greedy,
                temperature=None if greedy else e["temperature"],
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, batch["input_ids"].shape[1] :]
        texts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        log.info("generated %d / %d", len(texts), len(prompts))
    return texts


def score(records: list[dict]) -> dict:
    n = len(records)
    strict = sum(r["correct_strict"] for r in records)
    flexible = sum(r["correct_flexible"] for r in records)
    no_answer = sum(r["pred_strict"] is None for r in records)
    return {
        "n": n,
        "accuracy": round(100 * strict / n, 2) if n else 0.0,
        "accuracy_flexible": round(100 * flexible / n, 2) if n else 0.0,
        "no_final_answer_rate": round(100 * no_answer / n, 2) if n else 0.0,
        "mean_completion_chars": round(sum(len(r["completion"]) for r in records) / max(n, 1), 1),
    }


def evaluate(cfg: dict, model_path: str, tag: str) -> dict:
    from transformers import AutoTokenizer

    set_seed(cfg["seed"])
    ds = load_eval(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in ds["prompt"]
    ]

    use_vllm = cfg["eval"].get("use_vllm") and _vllm_available()
    log.info("Evaluating %s (%s) on %d problems with %s", tag, model_path, len(prompts),
             "vLLM" if use_vllm else "HF generate")
    start = time.time()
    if use_vllm:
        completions = generate_vllm(model_path, prompts, cfg)
    else:
        completions = generate_hf(model_path, prompts, tokenizer, cfg)
    wall = time.time() - start

    records = []
    for q, gold, comp in zip(ds["question"], ds["answer"], completions, strict=True):
        pred_s = extract_answer(comp, "strict")
        pred_f = extract_answer(comp, "flexible")
        records.append({
            "question": q,
            "gold": gold,
            "completion": comp,
            "pred_strict": pred_s,
            "pred_flexible": pred_f,
            "correct_strict": pred_s == gold,
            "correct_flexible": pred_f == gold,
        })

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"eval_{tag}.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "tag": tag,
        "model": model_path,
        "split": cfg["data"]["eval_split"],
        "decoding": {"temperature": cfg["eval"]["temperature"],
                     "max_new_tokens": cfg["eval"]["max_new_tokens"],
                     "backend": "vllm" if use_vllm else "hf"},
        **score(records),
        "eval_wall_seconds": round(wall, 1),
        "config_hash": config_hash(cfg),
        "env": env_info(),
    }
    write_json(results_dir / f"eval_{tag}.json", summary)
    log.info("%s: strict acc %.2f%% (flexible %.2f%%) on n=%d",
             tag, summary["accuracy"], summary["accuracy_flexible"], summary["n"])
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True, help="label for the result files, e.g. base")
    parser.add_argument("--model", default=None, help="model path/id (default: config base model)")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.overrides)
    evaluate(cfg, args.model or cfg["model"]["name_or_path"], args.tag)


if __name__ == "__main__":
    main()

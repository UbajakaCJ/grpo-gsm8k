"""GSM8K loading and prompt construction (shared by training and evaluation)."""

from __future__ import annotations

from .rewards import gold_answer


def build_messages(question: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": question.strip()},
    ]


def to_example(row: dict, system_prompt: str) -> dict:
    """Map a raw GSM8K row to the conversational format GRPOTrainer expects."""
    return {
        "prompt": build_messages(row["question"], system_prompt),
        "answer": gold_answer(row["answer"]),
        "question": row["question"],
    }


def load_split(cfg: dict, split_key: str, limit_key: str):
    """Load a GSM8K split as a ``datasets.Dataset`` with ``prompt``/``answer`` columns."""
    from datasets import load_dataset

    dcfg = cfg["data"]
    ds = load_dataset(dcfg["dataset"], dcfg.get("config"), split=dcfg[split_key])
    limit = dcfg.get(limit_key)
    if limit:
        ds = ds.select(range(min(int(limit), len(ds))))
    system_prompt = cfg["prompt"]["system"]
    return ds.map(
        lambda row: to_example(row, system_prompt),
        remove_columns=ds.column_names,
        desc=f"format {dcfg[split_key]}",
    )


def load_train(cfg: dict):
    return load_split(cfg, "train_split", "max_train_samples").shuffle(seed=cfg["seed"])


def load_eval(cfg: dict):
    return load_split(cfg, "eval_split", "max_eval_samples")

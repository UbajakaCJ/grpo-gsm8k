from pathlib import Path

import pytest

from grpo_gsm8k.config import apply_overrides, config_hash, deep_merge, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_main_config_is_consistent():
    cfg = load_config(CONFIGS / "colab_a100.yaml")
    t = cfg["train"]
    completions_per_step = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]
    # TRL requires the generation batch to split evenly into groups.
    assert completions_per_step % t["num_generations"] == 0
    assert cfg["model"]["name_or_path"] == "Qwen/Qwen2.5-1.5B"
    assert cfg["reward"]["correctness_weight"] > 0


def test_smoke_inherits_and_overrides():
    cfg = load_config(CONFIGS / "smoke.yaml")
    assert cfg["train"]["max_steps"] == 4
    assert cfg["train"]["learning_rate"] == 1.0e-6  # inherited
    assert cfg["prompt"]["system"]  # inherited
    t = cfg["train"]
    assert (t["per_device_train_batch_size"] * t["gradient_accumulation_steps"]) % t[
        "num_generations"
    ] == 0


def test_cli_overrides_parse_types():
    cfg = load_config(CONFIGS / "smoke.yaml", ["train.max_steps=10", "train.use_vllm=true",
                                              "hub.repo_id=me/model", "data.max_eval_samples=null"])
    assert cfg["train"]["max_steps"] == 10
    assert cfg["train"]["use_vllm"] is True
    assert cfg["hub"]["repo_id"] == "me/model"
    assert cfg["data"]["max_eval_samples"] is None


def test_bad_override():
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}} and base["a"]["b"] == 1


def test_config_hash_ignores_private_keys():
    assert config_hash({"x": 1, "_config_path": "a"}) == config_hash({"x": 1, "_config_path": "b"})
    assert config_hash({"x": 1}) != config_hash({"x": 2})

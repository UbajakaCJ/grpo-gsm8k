"""YAML config loading with single inheritance and dotted CLI overrides.

    cfg = load_config("configs/smoke.yaml", ["train.max_steps=10", "seed=1"])
    cfg["train"]["max_steps"]  # -> 10
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_scalar(raw: str) -> Any:
    # YAML parsing gives us ints, floats, bools, null and lists for free.
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {dotted}: {key} is not a mapping")
        node[keys[-1]] = _parse_scalar(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    cfg = apply_overrides(cfg, overrides)
    cfg["_config_path"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable short hash of the experiment-defining parts of a config."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]

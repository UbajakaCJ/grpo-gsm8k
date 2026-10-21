"""Reproducibility and run-metadata helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        )
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def env_info() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu_name(),
        "git_commit": git_commit(),
    }
    for pkg in ("torch", "transformers", "trl", "datasets", "peft", "vllm"):
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info[pkg] = None
    return info


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    return path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def dtype_kwarg(dtype) -> dict:
    """``from_pretrained`` dtype kwarg: ``dtype`` on transformers>=4.56, else ``torch_dtype``."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}

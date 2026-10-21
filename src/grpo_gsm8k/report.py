"""Turn eval/train artifacts into the headline result, a summary JSON, plots and README.

    python -m grpo_gsm8k.report --config configs/colab_a100.yaml

Reads   <results_dir>/eval_base.json, eval_trained.json(+ .jsonl), train_meta.json, train_log.json
Writes  <results_dir>/summary.json, headline.md, training_curves.png
Updates README.md between the RESULTS markers (only for the main, non-smoke run).
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import re
import subprocess
from pathlib import Path

from .config import load_config
from .utils import read_json, write_json

log = logging.getLogger("grpo_gsm8k.report")

DEFAULT_REPO = "github.com/UbajakaCJ/grpo-gsm8k"
README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


# ----------------------------------------------------------------------------- stats


def mcnemar_p(base: list[bool], trained: list[bool]) -> float:
    """Exact two-sided McNemar test on paired per-problem correctness."""
    b = sum(1 for x, y in zip(base, trained, strict=True) if x and not y)  # lost
    c = sum(1 for x, y in zip(base, trained, strict=True) if y and not x)  # gained
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def bootstrap_delta_ci(base: list[bool], trained: list[bool], iters: int = 2000,
                       seed: int = 0) -> tuple[float, float]:
    """95% paired-bootstrap CI for (trained - base) accuracy, in percentage points."""
    rng = random.Random(seed)
    n = len(base)
    diffs = [int(t) - int(b) for b, t in zip(base, trained, strict=True)]
    stats = sorted(100 * sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return round(stats[int(0.025 * iters)], 2), round(stats[int(0.975 * iters) - 1], 2)


def _load_correct(path: Path) -> list[bool]:
    import json

    return [json.loads(line)["correct_strict"] for line in path.read_text().splitlines() if line]


# ----------------------------------------------------------------------------- text


def repo_url() -> str:
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return DEFAULT_REPO
    m = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
    return f"github.com/{m.group(1)}" if m else DEFAULT_REPO


def hardware_phrase(gpu: str | None) -> str:
    if gpu and "A100" in gpu:
        return "one Colab A100"
    return f"one {gpu}" if gpu else "one GPU"


def model_display(name: str) -> str:
    return name.rstrip("/").split("/")[-1]


def headline(summary: dict) -> str:
    return (
        f"Ran a GRPO baseline on {model_display(summary['base_model'])} with exact-match math "
        f"rewards: GSM8K accuracy went from {summary['base_accuracy']:.1f}% to "
        f"{summary['trained_accuracy']:.1f}% after {summary['steps']} training steps on "
        f"{hardware_phrase(summary['gpu'])}. Code: {summary['repo']}"
    )


def results_markdown(summary: dict, plot_name: str | None) -> str:
    s = summary
    lines = [
        f"> {headline(s)}",
        "",
        "| | Base | GRPO | Δ |",
        "|---|---:|---:|---:|",
        f"| GSM8K test accuracy (strict, n={s['n_eval']}) | {s['base_accuracy']:.1f}% | "
        f"{s['trained_accuracy']:.1f}% | {s['delta_pp']:+.1f} pp |",
        f"| Accuracy, flexible extraction | {s['base_accuracy_flexible']:.1f}% | "
        f"{s['trained_accuracy_flexible']:.1f}% | "
        f"{s['trained_accuracy_flexible'] - s['base_accuracy_flexible']:+.1f} pp |",
        f"| No final answer emitted | {s['base_no_answer_rate']:.1f}% | "
        f"{s['trained_no_answer_rate']:.1f}% | |",
        "",
    ]
    if s.get("delta_ci95"):
        lo, hi = s["delta_ci95"]
        lines.append(
            f"Paired 95% bootstrap CI on Δ: [{lo:+.1f}, {hi:+.1f}] pp · exact McNemar "
            f"p = {s['mcnemar_p']:.2g} ({s['gained']} problems gained, {s['lost']} lost)."
        )
        lines.append("")
    lines.append(
        f"Setup: {s['steps']} steps × {s['prompts_per_step']} prompts × {s['num_generations']} "
        f"generations, greedy zero-shot eval with the training prompt, {s['gpu'] or 'GPU'}, "
        f"train time {s['train_minutes']:.0f} min, config hash `{s['config_hash']}`."
    )
    if plot_name:
        lines += ["", f"![training curves](results/{plot_name})"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- plot


def plot_curves(log_history: list[dict], out: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plot")
        return False

    rows = [r for r in log_history if "step" in r and "reward" in r]
    if not rows:
        return False
    steps = [r["step"] for r in rows]
    panels = [
        ("reward", "mean exact-match reward"),
        ("completions/mean_length", "mean completion length (tokens)"),
    ]
    panels = [(k, t) for k, t in panels if all(k in r for r in rows)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.6))
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (key, title) in zip(axes, panels, strict=True):
        ax.plot(steps, [r[key] for r in rows], color="#2f6db5", linewidth=1.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------- main


def build_summary(results_dir: Path) -> dict:
    base = read_json(results_dir / "eval_base.json")
    trained = read_json(results_dir / "eval_trained.json")
    meta = read_json(results_dir / "train_meta.json")
    summary = {
        "headline": None,
        "base_model": meta["base_model"],
        "steps": meta["global_steps"],
        "prompts_per_step": meta["prompts_per_step"],
        "num_generations": meta["num_generations"],
        "gpu": meta["env"].get("gpu"),
        "train_minutes": meta["train_wall_seconds"] / 60,
        "config_hash": meta["config_hash"],
        "repo": repo_url(),
        "n_eval": trained["n"],
        "base_accuracy": base["accuracy"],
        "trained_accuracy": trained["accuracy"],
        "delta_pp": round(trained["accuracy"] - base["accuracy"], 2),
        "base_accuracy_flexible": base["accuracy_flexible"],
        "trained_accuracy_flexible": trained["accuracy_flexible"],
        "base_no_answer_rate": base["no_final_answer_rate"],
        "trained_no_answer_rate": trained["no_final_answer_rate"],
    }
    base_jsonl, trained_jsonl = results_dir / "eval_base.jsonl", results_dir / "eval_trained.jsonl"
    if base_jsonl.exists() and trained_jsonl.exists():
        b, t = _load_correct(base_jsonl), _load_correct(trained_jsonl)
        if len(b) == len(t):
            summary["gained"] = sum(1 for x, y in zip(b, t, strict=True) if y and not x)
            summary["lost"] = sum(1 for x, y in zip(b, t, strict=True) if x and not y)
            summary["mcnemar_p"] = mcnemar_p(b, t)
            summary["delta_ci95"] = bootstrap_delta_ci(b, t)
    summary["headline"] = headline(summary)
    return summary


def update_readme(readme: Path, block: str) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text()
    if README_START not in text or README_END not in text:
        return False
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    readme.write_text(pattern.sub(f"{README_START}\n{block}\n{README_END}", text))
    return True


def report(cfg: dict, update_readme_file: bool = True) -> dict:
    results_dir = Path(cfg["results_dir"])
    summary = build_summary(results_dir)
    log_path = results_dir / "train_log.json"
    plot_name = "training_curves.png"
    has_plot = log_path.exists() and plot_curves(read_json(log_path), results_dir / plot_name)
    write_json(results_dir / "summary.json", summary)
    block = results_markdown(summary, plot_name if has_plot else None)
    (results_dir / "headline.md").write_text(block + "\n")
    # Only the main run (results/ at repo root) owns the README table.
    if update_readme_file and results_dir.resolve() == Path("results").resolve():
        if update_readme(Path("README.md"), block):
            log.info("README.md results section updated")
    print("\n" + summary["headline"] + "\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-readme", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report(load_config(args.config, args.overrides), update_readme_file=not args.no_readme)


if __name__ == "__main__":
    main()

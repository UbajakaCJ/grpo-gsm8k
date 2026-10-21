import json

from grpo_gsm8k.report import (
    bootstrap_delta_ci,
    build_summary,
    headline,
    mcnemar_p,
    report,
    update_readme,
)
from grpo_gsm8k.utils import write_json


def _write_fake_run(d, base_correct, trained_correct, gpu="NVIDIA A100-SXM4-40GB"):
    for tag, correct in (("base", base_correct), ("trained", trained_correct)):
        n = len(correct)
        acc = round(100 * sum(correct) / n, 2)
        write_json(d / f"eval_{tag}.json", {
            "n": n, "accuracy": acc, "accuracy_flexible": acc + 1,
            "no_final_answer_rate": 5.0,
        })
        with (d / f"eval_{tag}.jsonl").open("w") as f:
            for c in correct:
                f.write(json.dumps({"correct_strict": c}) + "\n")
    write_json(d / "train_meta.json", {
        "base_model": "Qwen/Qwen2.5-1.5B", "global_steps": 500, "prompts_per_step": 8,
        "num_generations": 8, "train_wall_seconds": 7200, "config_hash": "abc123",
        "env": {"gpu": gpu},
    })
    write_json(d / "train_log.json", [{"step": s, "reward": s / 10} for s in range(1, 6)])


def test_headline_matches_requested_template(tmp_path):
    _write_fake_run(tmp_path, [True] * 60 + [False] * 40, [True] * 72 + [False] * 28)
    s = build_summary(tmp_path)
    assert s["base_accuracy"] == 60.0 and s["trained_accuracy"] == 72.0
    assert s["gained"] == 12 and s["lost"] == 0
    assert s["headline"].startswith(
        "Ran a GRPO baseline on Qwen2.5-1.5B with exact-match math rewards: "
        "GSM8K accuracy went from 60.0% to 72.0% after 500 training steps on one Colab A100. "
        "Code: github.com/"
    )


def test_headline_names_other_gpus():
    s = {"base_model": "Qwen/Qwen2.5-1.5B", "base_accuracy": 1, "trained_accuracy": 2,
         "steps": 3, "gpu": "NVIDIA L4", "repo": "github.com/x/y"}
    assert "on one NVIDIA L4." in headline(s)


def test_mcnemar():
    assert mcnemar_p([True, False], [True, False]) == 1.0
    # 20 gained, 0 lost is highly significant
    assert mcnemar_p([False] * 20, [True] * 20) < 1e-5


def test_bootstrap_ci_brackets_delta():
    base = [True] * 50 + [False] * 50
    trained = [True] * 70 + [False] * 30
    lo, hi = bootstrap_delta_ci(base, trained)
    assert lo <= 20 <= hi


def test_update_readme_between_markers(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("top\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nbottom\n")
    assert update_readme(readme, "NEW")
    assert readme.read_text() == "top\n<!-- RESULTS:START -->\nNEW\n<!-- RESULTS:END -->\nbottom\n"


def test_report_end_to_end(tmp_path):
    _write_fake_run(tmp_path, [False, True, False, True], [True, True, True, False])
    s = report({"results_dir": str(tmp_path)}, update_readme_file=False)
    assert (tmp_path / "summary.json").exists()
    assert "Ran a GRPO baseline" in (tmp_path / "headline.md").read_text()
    assert s["delta_pp"] == 25.0

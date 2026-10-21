"""Answer extraction and exact-match reward functions for GSM8K.

Two extraction modes:

* ``strict``  - only an explicit final-answer marker counts: the last ``#### x``
  line or the last ``\\boxed{x}``. This is what the reward and the headline
  accuracy use, so the model is rewarded for *committing* to an answer.
* ``flexible`` - falls back to "the answer is x" and then to the last number in
  the text. Reported alongside strict accuracy as a diagnostic only.

Reward functions follow the TRL ``GRPOTrainer`` signature:
``fn(prompts, completions, **dataset_columns) -> list[float]``.
"""

from __future__ import annotations

import re
from fractions import Fraction

_NUMBER = r"-?\$?\s*\d[\d,]*(?:\.\d+)?%?|-?\$?\s*\.\d+%?"
_HASH_RE = re.compile(r"####\s*(" + _NUMBER + r"|[^\n]+)")
_BOXED_RE = re.compile(r"\\boxed\s*\{")
_ANSWER_IS_RE = re.compile(
    r"(?:final answer|the answer)\s*(?:is|:|=)\s*\**\s*(" + _NUMBER + r")", re.IGNORECASE
)
_ANY_NUMBER_RE = re.compile(_NUMBER)


def _last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, handling nested braces."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def normalize_number(raw: str | None) -> str | None:
    """Canonical string for a numeric answer, or None if it isn't numeric.

    ``"$1,234.50"`` -> ``"1234.5"``, ``"18."`` -> ``"18"``, ``"\\frac{3}{4}"`` -> ``"0.75"``.
    """
    if raw is None:
        return None
    s = raw.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("\\$", "").replace("$", "").replace(",", "").replace("\\!", "")
    s = s.replace("\\%", "").replace("%", "").strip()
    frac = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if frac:
        s = f"{frac.group(1)}/{frac.group(2)}"
    # Take the first number-looking token (drops trailing units like "dollars").
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?|-?\.\d+", s)
    if not m:
        return None
    token = m.group(0)
    try:
        value = Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None
    if value.denominator == 1:
        return str(value.numerator)
    as_float = float(value)
    return f"{as_float:.6f}".rstrip("0").rstrip(".")


def extract_answer(text: str, mode: str = "strict") -> str | None:
    """Extract and normalize the final answer from a model completion."""
    if mode not in {"strict", "flexible"}:
        raise ValueError(f"unknown mode {mode!r}")
    if not text:
        return None

    hashes = _HASH_RE.findall(text)
    if hashes:
        ans = normalize_number(hashes[-1])
        if ans is not None:
            return ans
    boxed = _last_boxed(text)
    if boxed is not None:
        ans = normalize_number(boxed)
        if ans is not None:
            return ans
    if mode == "strict":
        return None

    stated = _ANSWER_IS_RE.findall(text)
    if stated:
        ans = normalize_number(stated[-1])
        if ans is not None:
            return ans
    numbers = _ANY_NUMBER_RE.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def gold_answer(solution: str) -> str:
    """Normalized gold answer from a GSM8K reference solution (``... #### 72``)."""
    ans = normalize_number(solution.split("####")[-1])
    if ans is None:
        raise ValueError(f"Could not parse gold answer from: {solution[-80:]!r}")
    return ans


def is_correct(completion: str, gold: str, mode: str = "strict") -> bool:
    pred = extract_answer(completion, mode=mode)
    return pred is not None and pred == normalize_number(gold)


def completion_text(completion) -> str:
    """TRL passes plain strings or chat messages depending on the dataset format."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return "".join(m.get("content", "") for m in completion if m.get("role") == "assistant")
    return str(completion)


# --------------------------------------------------------------------------------------
# TRL reward functions. Dataset column ``answer`` holds the normalized gold answer.
# --------------------------------------------------------------------------------------


def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    """1.0 when the strictly-extracted final answer exactly matches gold, else 0.0."""
    return [
        1.0 if is_correct(completion_text(c), gold, mode="strict") else 0.0
        for c, gold in zip(completions, answer, strict=True)
    ]


_FORMAT_RE = re.compile(r"^####\s*\S+\s*$", re.MULTILINE)


def format_reward(prompts, completions, **kwargs) -> list[float]:
    """1.0 when the completion ends with exactly one well-formed ``#### x`` line."""
    scores = []
    for c in completions:
        text = completion_text(c).strip()
        lines = _FORMAT_RE.findall(text)
        ok = len(lines) == 1 and text.endswith(lines[0].strip())
        scores.append(1.0 if ok else 0.0)
    return scores

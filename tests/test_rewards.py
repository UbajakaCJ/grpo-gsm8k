import pytest

from grpo_gsm8k.rewards import (
    completion_text,
    correctness_reward,
    extract_answer,
    format_reward,
    gold_answer,
    is_correct,
    normalize_number,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("72", "72"),
        ("1,234", "1234"),
        ("$18.00", "18"),
        ("18.", "18"),
        ("0.50", "0.5"),
        ("-3", "-3"),
        ("45%", "45"),
        ("\\frac{3}{4}", "0.75"),
        ("\\$5", "5"),
        ("12 dollars", "12"),
        ("abc", None),
        (None, None),
    ],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_gold_answer_from_gsm8k_solution():
    sol = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n48+24 = <<48+24=72>>72\n#### 72"
    assert gold_answer(sol) == "72"
    assert gold_answer("blah\n#### 1,000") == "1000"


def test_strict_prefers_last_hash_line():
    text = "First guess #### 5\nWait, recompute: 3+4 = 7\n#### 7"
    assert extract_answer(text, "strict") == "7"


def test_strict_accepts_boxed_and_nested_braces():
    assert extract_answer("so the answer is \\boxed{42}", "strict") == "42"
    assert extract_answer("\\boxed{\\frac{1}{2}}", "strict") == "0.5"


def test_strict_rejects_unmarked_answer_but_flexible_accepts():
    text = "She has 3 apples and buys 4, so the answer is 7."
    assert extract_answer(text, "strict") is None
    assert extract_answer(text, "flexible") == "7"
    assert extract_answer("I computed 10 then 12", "flexible") == "12"


def test_empty_and_bad_mode():
    assert extract_answer("", "strict") is None
    with pytest.raises(ValueError):
        extract_answer("x", "lenient")


def test_is_correct_tolerates_formatting():
    assert is_correct("... #### $1,200.00", "1200")
    assert not is_correct("... #### 1201", "1200")


def test_completion_text_handles_chat_format():
    msgs = [{"role": "assistant", "content": "#### 3"}]
    assert completion_text(msgs) == "#### 3"
    assert completion_text("plain") == "plain"


def test_correctness_reward_trl_signature():
    completions = [
        [{"role": "assistant", "content": "2+2=4\n#### 4"}],
        [{"role": "assistant", "content": "the answer is 4"}],  # unmarked -> 0
        [{"role": "assistant", "content": "#### 5"}],
    ]
    rewards = correctness_reward(prompts=[None] * 3, completions=completions, answer=["4"] * 3)
    assert rewards == [1.0, 0.0, 0.0]


def test_format_reward():
    good = "step one\nstep two\n#### 12"
    two_answers = "#### 3\nhmm\n#### 12"
    trailing = "#### 12\nand some extra text"
    assert format_reward(None, [good, two_answers, trailing]) == [1.0, 0.0, 0.0]

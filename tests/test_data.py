from grpo_gsm8k.data import to_example


def test_to_example_conversational_format():
    row = {"question": " How many? ", "answer": "work\n#### 1,500"}
    ex = to_example(row, "Be careful.")
    assert ex["answer"] == "1500"
    assert ex["prompt"] == [
        {"role": "system", "content": "Be careful."},
        {"role": "user", "content": "How many?"},
    ]

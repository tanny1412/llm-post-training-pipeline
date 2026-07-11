from datasets import load_dataset


def format_prompt(question: str, schema: str) -> str:
    return (
        f"### Task\n"
        f"Generate SQL to answer this question: {question}\n\n"
        f"### Schema\n{schema}\n\n"
        f"### Answer\n"
    )


def format_example(row: dict) -> dict:
    return {
        "text":       format_prompt(row["question"], row["context"]) + row["answer"],
        "prompt":     format_prompt(row["question"], row["context"]),
        "completion": row["answer"],
    }


def load_splits() -> dict:
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(format_example)

    train    = dataset.select(range(70200))
    sft_eval = dataset.select(range(70200, 75200))
    held_out = dataset.select(range(75200, 78000))

    return {"train": train, "sft_eval": sft_eval, "held_out": held_out}


def check_contamination(splits: dict) -> None:
    train_answers    = set(splits["train"]["answer"])
    sft_eval_answers = set(splits["sft_eval"]["answer"])
    held_out_answers = set(splits["held_out"]["answer"])

    train_overlap    = train_answers & held_out_answers
    sft_eval_overlap = sft_eval_answers & held_out_answers

    assert len(train_overlap) == 0, \
        f"Contamination: {len(train_overlap)} train examples found in held-out test"
    assert len(sft_eval_overlap) == 0, \
        f"Contamination: {len(sft_eval_overlap)} sft_eval examples found in held-out test"

    print("Contamination check passed.")


if __name__ == "__main__":
    splits = load_splits()
    check_contamination(splits)
    print(f"Train:    {len(splits['train'])} examples")
    print(f"SFT eval: {len(splits['sft_eval'])} examples")
    print(f"Held-out: {len(splits['held_out'])} examples")

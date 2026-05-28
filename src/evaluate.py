import sqlite3

from tqdm import tqdm

from src.data import format_prompt


def evaluate_model(model, tokenizer, dataset, n: int = 500) -> float:
    correct = 0

    for row in tqdm(dataset.select(range(n)), desc="Evaluating", total=n):
        prompt = format_prompt(row["question"], row["context"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_sql = generated.split("### Answer\n")[-1].strip()

        try:
            conn = sqlite3.connect(":memory:")
            conn.executescript(row["context"])
            pred_rows = set(conn.execute(pred_sql).fetchall())
            true_rows = set(conn.execute(row["answer"]).fetchall())
            if pred_rows == true_rows:
                correct += 1
        except Exception:
            pass
        finally:
            conn.close()

    return correct / n

import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data import load_splits
from src.evaluate import evaluate_model

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def load_base_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


if __name__ == "__main__":
    splits = load_splits()
    model, tokenizer = load_base_model()

    mlflow.set_experiment("sql-post-training")

    with mlflow.start_run(run_name="baseline"):
        acc = evaluate_model(model, tokenizer, splits["held_out"], n=500)

        mlflow.log_metric("exec_accuracy", acc)
        mlflow.log_param("stage", "baseline")
        mlflow.log_param("model_id", MODEL_ID)

        print(f"Baseline execution accuracy: {acc:.4f}")

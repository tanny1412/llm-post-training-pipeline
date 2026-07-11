import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

from src.data import load_splits, format_prompt
from src.evaluate import evaluate_model

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
SFT_ADAPTER_PATH = "checkpoints/sft-adapter"


def load_merged_sft_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()

    return model, tokenizer


def build_dpo_dataset(model, tokenizer, sft_eval):
    preference_pairs = []

    for row in sft_eval:
        prompt = format_prompt(row["question"], row["context"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_sql = generated.split("### Answer\n")[-1].strip()

        if pred_sql != row["answer"]:
            preference_pairs.append({
                "prompt":   prompt,
                "chosen":   row["answer"],
                "rejected": pred_sql,
            })

    return preference_pairs


if __name__ == "__main__":
    splits = load_splits()
    model, tokenizer = load_merged_sft_model()

    print("Building DPO preference pairs from SFT failures...")
    preference_pairs = build_dpo_dataset(model, tokenizer, splits["sft_eval"])
    print(f"Built {len(preference_pairs)} preference pairs")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    dpo_config = DPOConfig(
        output_dir="checkpoints/dpo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        bf16=True,
        beta=0.1,
        logging_steps=10,
        save_steps=100,
        report_to="none",
    )

    mlflow.set_experiment("sql-post-training")

    with mlflow.start_run(run_name="dpo"):
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config,
            train_dataset=preference_pairs,
            tokenizer=tokenizer,
        )
        trainer.train()

        model = model.merge_and_unload()
        model.save_pretrained("checkpoints/dpo-merged")
        tokenizer.save_pretrained("checkpoints/dpo-merged")

        acc = evaluate_model(model, tokenizer, splits["held_out"], n=500)

        mlflow.log_metric("exec_accuracy", acc)
        mlflow.log_param("stage", "dpo")
        mlflow.log_param("beta", 0.1)
        mlflow.log_param("dpo_pairs", len(preference_pairs))

        print(f"DPO execution accuracy: {acc:.4f}")

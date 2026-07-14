import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

from src.data import load_splits
from src.evaluate import evaluate_model

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def load_model_with_lora():
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

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


if __name__ == "__main__":
    splits = load_splits()
    model, tokenizer = load_model_with_lora()

    sft_config = SFTConfig(
        output_dir="/workspace/checkpoints/sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        max_length=512,
        completion_only_loss=True,
        report_to="none",
    )

    mlflow.set_experiment("sql-post-training")

    with mlflow.start_run(run_name="qlora-sft"):
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=splits["train"],
            eval_dataset=splits["sft_eval"],
            args=sft_config,
        )
        trainer.train()

        acc = evaluate_model(model, tokenizer, splits["sft_eval"], n=500)

        mlflow.log_metric("exec_accuracy", acc)
        mlflow.log_param("stage", "sft")
        mlflow.log_param("r", 16)
        mlflow.log_param("lora_alpha", 32)
        mlflow.log_param("epochs", 3)
        mlflow.log_param("learning_rate", 2e-4)

        model.save_pretrained("/workspace/checkpoints/sft-adapter")
        tokenizer.save_pretrained("/workspace/checkpoints/sft-adapter")

        print(f"SFT execution accuracy: {acc:.4f}")

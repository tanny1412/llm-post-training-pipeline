# LLM Post-Training Pipeline: SQL Generation

Fine-tuning Llama-3.1-8B-Instruct for SQL generation using a full post-training pipeline — QLoRA SFT → DPO alignment → AWQ quantization → vLLM serving.

**Dataset:** [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) (78K SQL examples)  
**Eval metric:** Execution accuracy — the generated SQL is run against an in-memory SQLite database and the result set is compared to the ground truth. Not BLEU, not perplexity — actual correctness.

---

## Results

| Stage | Execution Accuracy | Notes |
|---|---|---|
| Baseline (no fine-tuning) | 1.6% | Zero-shot on custom prompt format |
| + QLoRA SFT | **95.8%** | 3 epochs, 70K examples, 4h41m on RTX 5090 |
| + DPO Alignment | TBD | Self-generated preference pairs |
| + AWQ Quantization | TBD | Target <1pp accuracy drop |

*Results will be updated as training completes.*

---

## Pipeline

```
Base Llama-3.1-8B-Instruct
        │
        ▼
[Stage 1] QLoRA SFT          ← teach the model to generate SQL in our format
        │  70K examples, 3 epochs, LoRA r=16
        │  Metric: execution accuracy on sft_eval
        ▼
[Stage 2] DPO Alignment      ← refine: prefer correct SQL over the model's own mistakes
        │  Self-generated (prompt, chosen, rejected) pairs from SFT failures
        │  beta=0.1, 1 epoch
        │  Metric: execution accuracy on held_out
        ▼
[Stage 3] AWQ Quantization   ← compress: 16GB bfloat16 → ~2GB INT4
        │  Activation-aware weight quantization
        │  Metric: execution accuracy on held_out (confirm <1pp drop)
        ▼
[Stage 4] vLLM Serving       ← serve: PagedAttention, continuous batching
           Target: 2x throughput vs HuggingFace generate()
```

---

## Key Design Decisions

**Why execution accuracy and not BLEU/perplexity?**  
A query can be phrased many ways and still return the correct rows. BLEU penalizes correct alternatives. Execution accuracy measures what actually matters: does the SQL return the right data?

**Why three splits instead of two?**  
The SFT eval set is used to generate DPO preference pairs — every SFT failure becomes a rejected example. This contaminates the SFT eval set for DPO measurement. The held-out test set is never touched during any training or pair generation — it's the single honest measure across all stages.

**Why QLoRA?**  
The base model (Llama-3.1-8B) loaded in 4-bit NF4 uses ~5GB VRAM instead of ~16GB. LoRA adds small trainable adapter matrices (r=16) to attention projections only — 0.78% of parameters are trained. Together they enable fine-tuning on a single GPU with no accuracy compromise versus full fine-tuning.

**Why DPO over PPO/RLHF?**  
DPO is offline supervised learning on preference pairs — no reward model, no generation loop during training, no PPO instability. The preference pairs come from the model's own SFT failures: no human labeling required.

**Why AWQ over GPTQ?**  
Quantization error = Δweight × activation. AWQ identifies the ~1% of weights with the largest input activations and scales them before quantization so they occupy the 4-bit range more precisely. GPTQ treats all weights equally. AWQ gives lower accuracy degradation at the same compression ratio.

---

## Project Structure

```
post-training-llms/
├── src/
│   ├── data.py          # load dataset, format prompts, split, contamination check
│   ├── evaluate.py      # execution accuracy evaluator (runs SQL against SQLite)
│   ├── baseline.py      # measure pre-training accuracy
│   ├── train_sft.py     # QLoRA supervised fine-tuning
│   ├── train_dpo.py     # DPO alignment on self-generated preference pairs
│   └── quantize.py      # AWQ quantization + vLLM serving (coming)
├── checkpoints/
│   ├── sft-adapter/     # LoRA adapter after SFT (~200MB)
│   ├── dpo-merged/      # merged DPO model in bfloat16 (~16GB)
│   └── awq/             # AWQ quantized model (~2GB)
├── devlogs.md           # learning journal — every concept explained with interview answers
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Clone
git clone https://github.com/tanishkandivlikar/post-training-llms
cd post-training-llms

# Install dependencies
pip install -r requirements.txt

# Requires HuggingFace access to Llama-3.1-8B-Instruct
huggingface-cli login
```

**Hardware used:** NVIDIA RTX 5090 (32GB VRAM) on RunPod  
SFT training time: ~4.5 hours | DPO: ~20 minutes | AWQ: ~15 minutes

---

## Running the Pipeline

```bash
# 1. Verify dataset splits and contamination check
python -m src.data

# 2. Baseline — measure accuracy before any fine-tuning
python -m src.baseline

# 3. QLoRA SFT — supervised fine-tuning
python -m src.train_sft

# 4. DPO alignment — preference learning on SFT failures
python -m src.train_dpo

# 5. AWQ quantization + vLLM serving (coming)
python -m src.quantize
```

MLflow experiment tracking runs automatically. View results:
```bash
mlflow ui
```

---

## What I Learned

This project was built Socratically — every design decision was questioned before writing code. The [devlogs.md](devlogs.md) contains the full learning journal: every concept, interview-ready answers, and the reasoning behind each choice.

Key concepts covered: NF4 quantization, LoRA low-rank decomposition, instruction masking, DPO loss derivation, beta as KL penalty, AWQ activation-aware scaling, PagedAttention in vLLM, execution accuracy as an eval metric, three-split evaluation design, MLflow experiment tracking.

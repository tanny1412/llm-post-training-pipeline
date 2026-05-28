# Post-Training LLM — Dev Log & Learning Journal

**Project:** Fine-Tuning Llama-3.2-3B for SQL Generation  
**Pipeline:** QLoRA SFT → DPO Alignment → AWQ Quantization → vLLM Serving  
**Dataset:** b-mc2/sql-create-context (78,577 examples)

---

## How This Log Works
- Every question asked → recorded here
- Your answer → recorded here
- The teaching/correction → recorded here
- Interview-ready answer → recorded here (memorize these)

---

## Session 1 — May 26, 2026 — Foundations

---

### Q1: Why does Llama-3.2-3B only get ~18% execution accuracy on SQL out of the box?

**Your answer:**
> "Because it's not specialized on SQL data, so it can complete but it is not aligned to answer in SQL — and then fix the right SQL using DPO."

**What you got right:**
- ✅ Not specialized on SQL data — correct

**What to sharpen:**
- ⚠️ You conflated SFT and DPO. DPO does not teach the model to answer in SQL — that's SFT's job. DPO refines quality *after* the model already knows to answer in SQL.

**The precise explanation:**
- A base model is trained with **next-token prediction** on web text
- It learns to **autocomplete**, not **answer**
- It has no concept of task format — ask it a question and it might generate more question text
- It may produce syntactically valid SQL that is logically wrong for the given schema
- SFT fixes this by showing thousands of (question + schema → correct SQL) pairs

**🎤 Interview-ready answer:**
> "A base model is trained with next-token prediction on web text — it learns to autocomplete, not answer. It has no concept of task format. When you ask for SQL, it might generate more natural language, or syntactically valid SQL that is logically wrong for the schema. SFT solves this by teaching input→output behavior on domain-specific examples."

---

### Q2: After SFT gets us to 72%, what kind of mistakes remain? What is DPO specifically fixing?

**Your answer:**
> "It still doesn't understand the quality of correctness of SQL from wrong SQL."

**What you got right:**
- ✅ The core idea — model can't distinguish good vs bad outputs

**The precise explanation:**
- SFT trains on correct examples only — the model **never sees its own mistakes**
- It has no internal signal to prefer one output over another
- It becomes **confidently wrong** — SQL that looks right but is logically wrong
- DPO fixes this by showing **contrastive pairs** from the model's own failures:
  ```
  Prompt:   "How many singers do we have?" + schema
  Chosen:   SELECT count(*) FROM singer        ✅
  Rejected: SELECT name FROM singer            ❌
  ```
- DPO pushes probability **UP** toward chosen and **DOWN** away from rejected — simultaneously
- The rejected examples come from the model's own SFT mistakes — no human labeling needed

**🎤 Interview-ready answer:**
> "After SFT, the model knows the task format but has no preference signal between good and bad outputs — it was only trained on correct examples. DPO introduces contrastive learning: for each prompt where the SFT model failed, we create a (correct, wrong) pair. DPO pushes probability mass toward correct SQL and away from wrong SQL simultaneously. No reward model needed."

---

### Follow-up to Q2: Why does DPO not need a reward model, but RLHF does?

**Your answer:**
> "DPO learns directly from correct and incorrect examples and pushes the model towards it. RLHF like PPO needs a reward model to score outputs — high positive for right, low for wrong."

**What you got right:**
- ✅ Core mechanism correct — DPO is direct, RLHF needs a scorer
- ✅ Correctly identified PPO as the RLHF algorithm

**The precise explanation:**
- **RLHF (PPO) flow:** Model generates output → Reward model scores it → PPO uses that score to update weights. The reward model is a **separate neural network** that can be wrong, unstable, expensive.
- **DPO flow:** (chosen, rejected) pairs → Direct loss on the policy model itself → No scorer needed
- DPO mathematically **bakes the reward signal into the loss function** — derives an equivalent objective from preference pairs directly

**🎤 Interview-ready answer:**
> "RLHF with PPO requires a separate reward model trained on human preferences to score outputs at each step. That reward model is an extra component that can be unstable or mis-calibrated. DPO eliminates the reward model entirely — it derives an equivalent optimization objective directly from (chosen, rejected) pairs. The preference signal is baked into the loss function itself, making training simpler, more stable, and faster."

---

### Q3: Why QLoRA over full fine-tuning? What does Q do, what does LoRA do, why both?

**Your answer:**
> "Quantized means base model in 4-bit but adapters in full 16-bit which are being trained. LoRA — extra matrices around projection matrices, and we need both because adapter trains/learns and base model frozen, input flows and backprop gradient flows through adapters."

**What you got right:**
- ✅ Base model in 4-bit, adapters in bfloat16 — exactly right
- ✅ LoRA injects matrices into projection layers — correct
- ✅ Gradients only flow through adapters, base model frozen — correct

**The LoRA math (what interviewers dig into):**
```
Output = W·x  +  (B · A) · x
          ↑             ↑
     frozen 4-bit    trainable adapter (bfloat16)
```
- W is shape `4096 × 4096` = 16M params (frozen)
- A is shape `4096 × r`, B is shape `r × 4096` where r = rank (e.g. 16)
- A×B = same shape as W, but only `2 × 4096 × 16 = 131K params`
- Total trainable: ~24M params out of 3B = **0.79%**
- Why it works: weight changes needed for fine-tuning have **low intrinsic rank**

**Why you need BOTH Q + LoRA:**
| Problem | Solution |
|---|---|
| 3B model in FP16 = 6.4GB just to load | Q: Load in 4-bit NF4 = ~2GB |
| Can't backprop through 4-bit weights | LoRA: Backprop only through bfloat16 adapters |
| Gradient storage for 3B params = 6GB+ | LoRA: Only 24M adapter params need gradients |

**🎤 Interview-ready answer:**
> "QLoRA combines two ideas. First, quantization: the base model is loaded in 4-bit NF4, reducing VRAM from ~12GB to ~2GB. Second, LoRA: instead of updating frozen base weights, we inject small trainable matrices A and B into each projection layer. The update is ΔW = B·A where both have rank r, so we only train ~24M parameters instead of 3B. Gradients flow through the bfloat16 adapters only — never through the 4-bit base weights. Without quantization the model doesn't fit. Without LoRA you can't backprop through quantized weights. Together they enable full fine-tuning quality on a single consumer GPU."
### Q4: Why three splits instead of the standard train/test two-split?

**Your answer:**
> "SFT eval set was used to generate DPO pairs so it's contaminated, so we need a separate eval for DPO as well? Not held-out?"

**What you got right:**
- ✅ SFT eval is contaminated because failures became DPO rejected examples — exactly right

**The small correction:**
- The held-out test IS the clean measurement for DPO — you don't need a 4th split
- Three splits is enough:
```
SFT eval (5K)       → SFT hyperparameter decisions + source of DPO rejected pairs
                      CONTAMINATED for DPO final measurement

Held-out test (2.8K) → ONE honest final measurement of ALL stages at the very end
                       NEVER touched during any training
```
- The held-out test is your single source of truth — touched exactly ONCE, for final reporting

**🎤 Interview-ready answer:**
> "We need three splits because the SFT eval set is used to generate DPO preference pairs — every failure becomes a rejected example. This means the SFT eval set is contaminated with respect to DPO: the model has indirectly seen those examples during DPO training. Measuring DPO improvement on that set would give inflated numbers. The held-out test set is never touched during any training or DPO pair generation — it's the single honest measure for comparing all stages at the end."
### Q5: Why is AWQ smarter than GPTQ? Why do some weights matter more during quantization?

**Your answer:**
> "Some parameters have high signal activations that affect more because of input — so those are high-performing params, we can let go of activations which don't change or matter."

**What you got right:**
- ✅ Core intuition correct — weights with large activations are more important
- ✅ Correctly identified that low-activation weights can be safely rounded aggressively

**The precise mechanism:**
```
output = weight × activation
error  = Δweight × activation
```
- Large activation → small rounding error becomes a **big output error** (salient weight)
- Small activation → same rounding error barely matters

**GPTQ vs AWQ:**
- **GPTQ:** Rounds all weights to 4-bit equally — fast but naive
- **AWQ:** 
  1. Runs a calibration dataset to find ~1% of weights with consistently large input activations
  2. Scales those weights UP before quantization → scales activations DOWN proportionally
  3. Quantizes everything to 4-bit
  4. Important weights now use the 4-bit range more precisely → less rounding error where it matters

**🎤 Interview-ready answer:**
> "Quantization error is Δweight × activation. A small rounding error on a weight with large activations causes a large output error — those weights are salient. GPTQ treats all weights equally. AWQ identifies the ~1% of weights with the highest activation magnitudes using a calibration dataset, then scales those weights before quantization so they occupy the 4-bit range more precisely. The result is the same model size but with accuracy-critical weights protected — hence less than 1% accuracy drop versus GPTQ's higher degradation."

---

### Side Q: Why does DPO only train for 1 epoch, not 3 like SFT?

**Three reasons:**
1. **Tiny dataset** — only ~4K pairs vs 70K for SFT. More epochs = overfitting
2. **Refinement not education** — model already knows SQL. DPO just nudges probability mass. One pass is enough
3. **DPO can collapse** — overtraining causes the model to refuse generating anything except exact chosen examples. KL divergence penalty helps but 1 epoch is the safe default

**On eval during DPO:** DPO runs so fast (20 min, 1 epoch) that you train once then evaluate on held-out test after. No DPO-specific eval loop needed.

---

### LoRA Deep Dive — Projection Matrices, Injection, and Merging

**What are projection matrices?**
- Wq, Wk, Wv, Wo — the weight matrices inside attention that transform input embeddings
- Each roughly `4096 × 4096` = 16M parameters in Llama-3.2-3B
- Called "projection" because they project input into a new space (query/key/value space)

**What does "inject" mean?**
```
Without LoRA:  output = Wq · x
With LoRA:     output = Wq · x  +  (B · A) · x
```
- A: `4096 × 16` = 65,536 params
- B: `16 × 4096` = 65,536 params
- Total per matrix: 131K params vs 16M for full Wq — **125x fewer**
- B·A adds a low-rank correction on top of the frozen weight

**Why low rank works:**
- Weight changes needed for fine-tuning have low intrinsic rank
- The model doesn't need to update all 16M dimensions — just a low-dimensional subspace

**Merge vs not merge:**
- **During serving:** merge — `Wq_merged = Wq + B·A` once, then single computation every forward pass
- **Before DPO:** merge SFT adapter first — the merged model becomes the frozen DPO reference

**Does DPO also use LoRA?**
Yes — same LoRA infrastructure, different loss function:
- SFT loss: next token prediction on correct examples
- DPO loss: preference between chosen and rejected + KL divergence from reference

**The full adapter flow:**
```
Base (frozen)
  + SFT adapter → train 3 epochs → merge
                                      ↓
                              SFT merged model = DPO reference (frozen)
                                      ↓
                              + DPO adapter → train 1 epoch
                                      ↓
                              DPO loss = preference loss + KL(current || SFT reference)
                                      ↓
                              merge → final model → AWQ quantization
```

**Why start a fresh DPO adapter instead of continuing the SFT adapter?**
- DPO needs a reference model that never changes during training
- Merging SFT adapter first creates a clean frozen reference
- If you continued the same adapter, the reference would shift during DPO — you'd lose the anchor

**🎤 Interview-ready answer:**
> "After SFT, I merge the LoRA adapter into the base model weights to get a clean SFT model. This merged model becomes the frozen reference for DPO. I then attach a fresh LoRA adapter and train with DPO loss — which has two components: a preference term pushing probability toward correct SQL over wrong SQL, and a KL divergence term preventing the model from drifting too far from the SFT reference. After DPO, I merge again before AWQ quantization."

**Clarification added later:**
- DPO injects LoRA into the same projection matrices (Wq, Wk, Wv, Wo) of the merged SFT model
- Same architecture as SFT LoRA, different loss function
- SFT adapter: trained on "here is the correct answer"
- DPO adapter: trained on "this answer is better than that answer"

### Important: What format is the merged model in?

**Question:** Can you add 4-bit NF4 weights + bfloat16 adapter directly?

**Your answer:** "I don't think you can add directly"

**Correct — here's what actually happens:**
```
4-bit NF4 base → dequantize → bfloat16
                                  +
                         bfloat16 adapter
                                  =
                         bfloat16 merged model
```

**Full format flow across all stages:**
```
SFT Training:
  Base model: NF4 4-bit  ← loaded this way to fit in VRAM
  Adapter:    bfloat16   ← trains on top

Merge for DPO:
  Dequantize: NF4 → bfloat16
  Add adapter: bfloat16 + bfloat16
  Result: bfloat16 merged model (~6.4GB)

DPO Training:
  Base model: bfloat16 merged SFT model (frozen reference)
  New adapter: bfloat16 ← trains on top

AWQ Quantization:
  bfloat16 merged DPO model → AWQ → INT4 (~2.1GB)
```

**Key insight:** DPO trains on a REAL bfloat16 model — not the quantized one. The 4-bit loading was only to make SFT fit in VRAM during training.

---

### Follow-up: Why is PPO considered RL but DPO is not?

**Your answer:**
> "Policy update and reward model which DPO doesn't have"

**Sharpened:**
- Policy update happens in BOTH — that's not what makes PPO "RL"
- What makes PPO RL is the **closed generate→score→update loop**:
```
PPO: Model generates SQL → Reward model scores it dynamically → PPO updates → repeat (online)
DPO: Looks at pre-collected (chosen, rejected) pairs → single supervised pass (offline)
```
- PPO interacts with an environment at runtime. DPO is entirely offline.

**🎤 Interview-ready answer:**
> "RLHF with PPO is reinforcement learning because it has the RL loop — the model generates outputs, a reward model scores them at runtime, and PPO updates the policy based on those dynamic rewards. It's online and iterative. DPO is offline — it never generates during training. It takes pre-collected preference pairs and directly optimizes a closed-form loss. No reward model, no generation loop. DPO is technically supervised learning on preference data."

**Key phrase:** *DPO is supervised learning on preference data — not RL.*

---

## Evaluation Strategy — Beyond Execution Accuracy

These notes extend the original plan. Add these on top of the PDF's execution accuracy metric.

### 1. Task-Specific Benchmarks
- Don't just measure perplexity — define what "correct" means concretely
- For SQL: does the generated query return the right rows?
- Create 50-100 test cases with known correct answers, measure pass rate before and after each stage
- Our metric: **execution accuracy** (query runs + returns correct rows) — already in the plan ✅

### 2. Human Preference Alignment
- DPO is about preferences — so measure preferences directly
- Show Model A (SFT) vs Model B (SFT+DPO) response to 20 prompts
- Does DPO actually improve preference rate over SFT alone?
- For SQL this is verifiable automatically — correct rows = preferred

### 3. Regression Testing
- Does fine-tuning on SQL hurt general capability?
- Run **MMLU or HellaSwag** before and after fine-tuning
- Goal: gained domain skill WITHOUT losing general ability
- The story: "MMLU dropped only 0.3% — SQL accuracy went up 54pp"

### 4. LLM-as-Judge
- Use GPT-4 to evaluate responses at scale
- Prompt: *"On a scale of 1-10, how helpful is this SQL response? Does it answer the question correctly?"*
- Cheap, scalable, good for qualitative evaluation alongside execution accuracy

### The Story This Lets You Tell
> *"Base Llama scored 18% execution accuracy. After SFT it hit 72%. After DPO it hit 79%. General capability (MMLU) dropped only 0.3%. AWQ quantization cost less than 1pp accuracy but cut latency in half."*

**That is evaluation. That is what stands out on a resume and in interviews.**

---

### Pre-Code Q: Why does seed=42 matter when shuffling before splitting?

**Your answer:**
> "Then we get different splits leading to different eval scores."

**What you got right:**
- ✅ Core idea — different splits = different scores = incomparable results

**The precise explanation:**
```
Person A: easy queries in train, hard in test → model looks bad
Person B: hard queries in train, easy in test → model looks great
```
Same model, same code, different reported accuracy. Can't compare, can't reproduce.

With seed=42 → everyone gets identical splits → results reproducible → comparisons fair.

This is Rule 1 in the PDF for a reason — it's the foundation of scientific integrity.

**🎤 Interview-ready answer:**
> "Without a fixed seed, shuffling is non-deterministic. Different runs produce different train/eval/test splits — easy examples might land in test for one run and train for another, making accuracy numbers incomparable and irreproducible. Fixing seed=42 ensures every run produces identical splits, so results are reproducible and comparisons between stages are fair."

---

---

## Questions Never Asked — Filled In

### NF4 vs INT4 vs FP4 — Why NF4 specifically?

**Question:** Neural network weights — what distribution do they follow?
**Your answer:** "idk"

**The answer:**
- Neural network weights follow a **normal distribution** — bell curve, clustered around zero
- Almost all weights are between -2 and +2. Very few are large.

**Why this makes NF4 better than INT4:**
```
INT4: uniform grid → -8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7
      most slots (-8 to -3, 3 to 7) wasted — almost no weights live there

NF4: values spaced to match normal distribution
     more slots near zero (where weights cluster)
     fewer slots at extremes (where weights rarely go)
```
Same 4 bits. Precision goes where data actually is → lower quantization error.

**🎤 Interview-ready answer:**
> "Neural network weights follow a normal distribution — most are clustered near zero. INT4 uses a uniform grid, wasting precision at the extremes where few weights exist. NF4 — Normal Float 4 — places its 16 quantization levels according to the normal distribution, so precision is concentrated where weights actually cluster. This gives NF4 lower quantization error than INT4 for the same 4-bit budget."

### `lora_alpha` — What does it control?

**Question:** `r=16, lora_alpha=32` — what does alpha do? What if scale is too high or too low?

**Your answer:**
> "It's like a control for how much to change weights with LoRA. High = fine-tuning intensity is higher."

**What you got right:**
- ✅ Alpha controls the intensity of the adapter's contribution

**The precise mechanism:**
```
output = Wq·x + (alpha/r) × (B·A)·x

With r=16, alpha=32 → scale = 32/16 = 2.0
```

**Two failure modes:**
- **Scale too high (e.g. 10):** Adapter dominates → overwrites base model knowledge → catastrophic forgetting → loses general capability after merging
- **Scale too low (e.g. 0.01):** Adapter signal too weak → tiny gradients → model barely learns → underfitting

**Why `alpha = 2r` is the standard default:**
- Scale of 2.0 — adapter contributes meaningfully without overpowering base
- Not derived mathematically — empirical community standard
- Tuning rule: domain very different from pretraining → increase alpha. Domain close → decrease alpha.

**🎤 Interview-ready answer:**
> "`lora_alpha` controls the scaling of the adapter's contribution — the effective scale is `alpha/r`. Too high and the adapter overwrites the base model's pretrained knowledge, causing catastrophic forgetting. Too low and the adapter's signal is too weak to learn. Setting `alpha = 2r` (scale=2) is the standard default — it lets the adapter contribute meaningfully without dominating the base model. For SQL fine-tuning, where the domain is moderately different from pretraining, this is a reasonable starting point."

### `target_modules` — Why attention projections, not MLP?

**Question:** Why inject LoRA into q_proj, k_proj, v_proj, o_proj only — not the MLP layers?

**Your answer:**
> "It learns vocabulary embeddings so it will learn to answer in new vocab i.e. SQL language."

**What to sharpen:**
- ⚠️ Embedding layer handles vocabulary — attention does something different
- Attention learns *routing patterns* — which tokens to attend to
- MLP layers store *factual world knowledge* from pretraining

**The precise explanation:**
- SQL generation is a **routing problem**: attend to table names when generating FROM, column names when generating SELECT, question keywords when deciding COUNT vs SUM
- Attention projections (Wq, Wk, Wv, Wo) control which tokens influence which outputs — these routing patterns need to change for SQL
- MLP layers store world knowledge — we don't need new facts, we need new attention behaviour
- Targeting attention captures the behavioural change with fewest trainable parameters

**🎤 Interview-ready answer:**
> "We target attention projections because SQL generation is a routing problem — the model needs to learn to attend to schema elements (table names, column names) when generating SQL keywords. Attention weights control which tokens influence which outputs. The MLP layers store factual world knowledge from pretraining, which we don't need to change — we need to change how the model routes attention, not what facts it knows. Targeting q, k, v, o projections captures this behavioural change with the fewest trainable parameters."

### Double Quantization — What is `bnb_4bit_use_double_quant=True`?

**Question:** Scale constants for NF4 quantization — how much memory do they add, and what does double quantization do?

**The math:**
```
3,000,000,000 weights / 64 per block = 46,875,000 blocks
46,875,000 × 4 bytes (FP32) = ~187MB just for scale constants
```

**What double quantization does:**
```
Round 1: FP16 weights → NF4 (4-bit)
         scale constants: FP32 per 64 weights → 187MB

Round 2: FP32 scale constants → FP8 (8-bit)
         with their own scale constants: FP32 per 256 constants
         
Savings: 187MB × (8/32) = ~47MB saved
```
Free VRAM savings — no accuracy cost.

**🎤 Interview-ready answer:**
> "Quantization requires storing a scale constant per block of weights to enable dequantization. With 3B weights and block size 64, that's ~187MB of FP32 constants. Double quantization quantizes those constants themselves — from FP32 to FP8 — saving ~47MB with negligible accuracy impact. It's a second pass of quantization applied to the quantization metadata itself."

### PagedAttention — Skipped for now, revisit when we reach Stage 4 (vLLM serving)

### `beta` in DPO — What does it control?

**Question:** DPO loss = preference loss + beta × KL(current || reference). What if beta is too high or too low?

**Your answers:**
> "Beta very high → changes aligned model very different from SFT, SFT learning wasted"
> "Beta very low → DPO won't distinguish right from wrong SQL, stays similar to SFT"

**Wait — directions are flipped:**
- **High beta** → KL penalty strong → model anchored to SFT → can't drift → learns little from DPO
- **Low beta** → KL penalty weak → model drifts freely → forgets SFT learning → SQL patterns lost

```
beta too high (e.g. 10):   anchored to SFT → DPO learns nothing → stays at 72%
beta too low (e.g. 0.001): drifts from SFT → forgets SQL patterns → quality drops
beta = 0.1 (default):      balanced → learns preferences, preserves SFT foundation
```

**🎤 Interview-ready answer:**
> "`beta` controls the KL divergence penalty in DPO — how much the model is allowed to drift from the SFT reference. High beta anchors the model tightly to SFT, preventing DPO from learning preferences. Low beta lets the model drift too far, forgetting the SQL patterns SFT taught. `beta=0.1` is the standard default — loose enough to learn the chosen/rejected distinction, tight enough to preserve SFT's foundation."

**Memory device:** beta is a **leash length**.
- Short leash (high beta) = can't go far from SFT = can't learn preferences
- Long leash (low beta) = wanders too far from SFT = forgets SQL patterns

### `max_seq_length=512` — Why this number, what happens with truncation?

**Question:** What happens to examples longer than 512 tokens? Why not use 2048?

**Your answer:** "I don't know why truncate, and also why not 2048 — maybe because of inference?"

**What happens with truncation:**
- Long examples (e.g. 600 tokens) get cut to 512 — last 88 tokens dropped
- Usually drops end of schema OR the answer itself
- If answer is truncated → model trains on incomplete SQL → **silent data quality bug**
- Fix: truncate from schema side, never answer side. Or filter long examples entirely.

**Why not 2048:**
- Attention is quadratic in sequence length:
```
512  tokens → 512²  = 262,144 attention ops per layer
2048 tokens → 2048² = 4,194,304 attention ops per layer
4x longer sequence = 16x more memory
```
- On 24GB GPU: difference between batch size 4 working vs OOM crash
- Most SQL examples fit in 512 — setting 2048 wastes memory on padding

**🎤 Interview-ready answer:**
> "`max_seq_length=512` balances coverage vs memory. Attention complexity is quadratic — 4x longer sequence means 16x more memory. Most SQL examples in this dataset fit in 512 tokens, so going to 2048 wastes GPU memory on padding. The risk with truncation is cutting the answer rather than the schema — so the correct approach is to truncate from the schema side or filter long examples entirely, never train on truncated answers."

---

## Code Written So Far
_None yet — starting with src/data.py_

---

## `src/data.py` — Design Decisions

### Responsibilities (agreed before writing):
1. Load dataset from HuggingFace (b-mc2/sql-create-context)
2. Shuffle with seed=42
3. Split into 3 — train (70,200), sft_eval (5,000), held-out test (2,800)
4. `format_prompt(question, schema)` — prompt only, imported everywhere
5. `format_example(row)` — full example including answer, used for SFT training
6. Contamination check — assert no overlap between train answers and test answers

### Why instruction masking matters:
- SFT should only compute loss on answer tokens, not question + schema tokens
- If loss computed on question too → model optimizes for predicting questions, not SQL
- Called **instruction masking** — mask loss on instruction part, only learn the completion
- SFTTrainer handles this automatically with the correct format

### Why `format_prompt` and `format_example` are separate:
- `format_prompt` is used in 3 places: data.py, evaluate.py, baseline.py, train_dpo.py
- `format_example` is only used for SFT training (includes the answer)
- During evaluation, the model generates the answer — so you can't include it in the prompt
- If combined into one function, you'd have to duplicate the prompt format everywhere
- Rule 2 from PDF: "Define format_prompt() once. Import it everywhere. Never inline."

```python
# format_prompt — used everywhere (no answer)
def format_prompt(question, schema):
    return f"### Task\nGenerate SQL: {question}\n\n### Schema\n{schema}\n\n### Answer\n"

# format_example — only for SFT training (includes answer)
def format_example(row):
    return format_prompt(row['question'], row['context']) + row['answer']
```

### Key concepts from writing `check_contamination()`:

**Why use `answer` as fingerprint, not `question` or `text`:**
- Two different questions can have the same SQL answer — that's fine, not contamination
- What matters: did the model see this exact SQL query during training?
- `text` too strict — tiny formatting difference misses real duplicates
- `question` misses cases where same SQL appears with different phrasings
- `answer` = exact SQL query = the thing the model might have memorized

**Why `assert` and not just `print`:**
- Never silently continue with contamination — it corrupts every result you report
- Crash loudly with a clear message so the bug is impossible to miss

**How set intersection works:**
```python
train_answers = set(...)     # O(1) lookup, deduplicates within split
& held_out_answers           # intersection — elements in BOTH sets
= {}                         # empty = no contamination = good
```
- Faster than nested loops: O(min(n,m)) vs O(n*m)
- For 70K vs 2.8K examples the speed difference is huge

### Key concepts from writing `load_splits()` and `format_example()`:

**Why `format_example` returns a dict not a string:**
- `dataset.map()` needs a dict to know which column to add
- `{"text": ...}` → adds a column called `"text"` to every row
- Returning a string → map doesn't know what column to assign it to

**What `dataset.map(format_example)` does visually:**
```
Before: [question, context, answer]  per row
After:  [question, context, answer, text]  per row

text = "### Task\nGenerate SQL: {question}\n\n### Schema\n{context}\n\n### Answer\n{answer}"
```
Iterates every row, calls `format_example(row)`, adds the returned `"text"` key as a new column.

**Why map before splitting:**
- All three splits automatically get the `"text"` column
- One map call instead of three

**Why return a dict from `load_splits()`:**
- `splits["train"]` is more readable than positional unpacking
- Keys make it clear which split is which

**Why we ignore HuggingFace's own train/test split:**
- We control our own shuffle seed and split sizes
- Guarantees no leakage, reproducibility, exact sizes we need

**Why 78,000 not 78,577:**
- Clean round numbers, simplicity

### What runs as importable vs script:
**Importable functions** (other files need these):
- `format_prompt()` — evaluate.py, baseline.py, train_dpo.py
- `format_example()` — train_sft.py
- `load_splits()` — every training script

**Script behaviour** (only when `python src/data.py`):
- Contamination check — one-time sanity check
- Print split sizes — confirm numbers look right

```python
def load_splits():         # importable
    ...

def format_prompt():       # importable
    ...

if __name__ == "__main__": # only runs as script
    splits = load_splits()
    check_contamination(splits)
    print("Splits OK")
```

---

## Key Numbers to Know Cold (from PDF)

| Stage | Execution Accuracy | Latency (p50) | VRAM | Model Size |
|---|---|---|---|---|
| Base Llama-3.2-3B | ~18% | ~340ms | ~12GB | 6.4GB FP16 |
| + QLoRA SFT | ~72% (+54pp) | ~340ms | ~8GB | 6.4GB + 80MB adapter |
| + DPO Alignment | ~79% (+7pp) | ~340ms | ~8GB | 6.4GB + 80MB adapter |
| + AWQ + vLLM | ~78% (-1pp) | ~162ms (2.1x) | ~4GB | ~2.1GB INT4 |

---

## Clean Experiment Rules (Memorize These)

| # | Rule | Why |
|---|---|---|
| 1 | Shuffle before splitting | Prevents easy/hard query bias across splits |
| 2 | Identical prompt format everywhere | One extra newline = different distribution |
| 3 | Use eval set for decisions, test set for reporting | Test set peeking = soft label leak |
| 4 | Check for data contamination | Overlap inflates eval numbers |
| 5 | Log everything to MLflow from day one | You won't remember what hyperparams did what |
| 6 | Save checkpoints after every stage | Can't retrain if AWQ corrupts the model |
| 7 | Baseline first — record it before touching anything | No before = no story |

---

## Dataset Splits

| Split | Size | Purpose | Used in DPO? |
|---|---|---|---|
| SFT train | 70,200 | Train the SFT model | No |
| SFT eval | 5,000 | Measure SFT accuracy + generate DPO pairs from failures | Yes — source of rejected examples. Cannot use for final eval |
| Held-out test | 2,800 | Final honest evaluation of all stages | Never touched until the end |

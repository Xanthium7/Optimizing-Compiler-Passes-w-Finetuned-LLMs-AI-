# 🧠 Complete Explanation: Finetuning & Evaluation Code

Your project **finetunes an 8-billion-parameter LLM** to act as a **compiler optimizer** — it takes unoptimized LLVM IR and produces O3-optimized IR. Then it **evaluates** how good the model's optimizations are by measuring CPU cycle counts.

---

# Part 1: Finetuning — [mini_port_finetune_in_code.py](mini_port_finetune_in_code.py)

---

## 1. Loading the Base Model (Lines 8–17)

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B-Base",
    max_seq_length = 8192,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)
```

### Key Concepts:

### Unsloth
A library that makes finetuning LLMs **2–5x faster** and uses **70% less memory** versus vanilla HuggingFace. It achieves this via custom CUDA kernels and math optimizations. You use it as a drop-in replacement for loading and training models.

### Qwen3-8B-Base
The **base model** being finetuned. It has 8 billion parameters, made by Alibaba. "Base" means it was only pre-trained on general next-token prediction — it hasn't been instruction-tuned yet. Your finetuning teaches it the specific IR optimization task.

### Tokenizer
Converts text ↔ numbers (tokens). For example, `"hello"` → `[15339]`. Every model has its own tokenizer with its own vocabulary. You need the tokenizer to prepare inputs and decode outputs.

### `max_seq_length = 8192`
Max tokens the model can process at once. LLVM IR blocks can be very long, so 8192 gives enough room. Anything longer gets truncated (you filter these out later).

### `dtype = torch.bfloat16` — Brain Float 16

| Type | Bits | Range | Precision | Best For |
|---|---|---|---|---|
| `float32` | 32 | Huge | High | Default, slow |
| `float16` | 16 | Limited | Medium | T4 GPUs |
| **`bfloat16`** | **16** | **Same as float32** | **Lower** | **A100/H100 GPUs** |

`bfloat16` uses 16 bits but allocates more bits to the **exponent** (range) than the **mantissa** (decimal precision). This means it handles the same range of numbers as float32 but with less precision. A100/H100 GPUs have **hardware-native bfloat16 support**, so it's faster than float16 on those GPUs.

### `load_in_4bit = True` — The "Q" in QLoRA

This loads the model weights **quantized to 4 bits** instead of the original 16 bits:

| Precision | Memory for 8B model |
|---|---|
| float32 | ~32 GB |
| float16/bfloat16 | ~16 GB |
| **4-bit** | **~4–5 GB** |

> [!IMPORTANT]
> **Quantization** = compressing model weights by reducing their numerical precision. The specific format used is **NF4 (NormalFloat4)**, which distributes its 16 possible values (2⁴=16) along a bell curve to match the typical distribution of neural network weights, giving better precision where most weights actually are. Without 4-bit quantization, an 8B model wouldn't fit on a single GPU for finetuning.

---

## 2. Loading the Dataset (Lines 20–39)

```python
ds = load_dataset("YangziResearch/IR-OptSet", split="train")
selected = ds.select(range(20000))
train_test = selected.train_test_split(test_size=200, seed=42)
train_ds = train_test["train"]  # 19,800 samples
test_ds  = train_test["test"]   # 200 samples
```

The dataset `IR-OptSet` contains pairs of:
- **`preprocessed_ir`** — unoptimized LLVM IR (model input)
- **`o3_ir`** — O3-optimized LLVM IR (target output the model should learn to produce)

**`seed=42`** ensures the train/test split is **reproducible** — the same 200 samples always end up in the test set.

**`DRY_RUN`** — when `True`, uses 2,200 samples for quick testing. When `False`, uses the full 20,000.

---

## 3. The Alpaca Prompt Template (Lines 42–62)

```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input...

### Instruction:
{}

### Input:
{}

### Response:
{}"""
```

### Concept: Alpaca Prompt Format

When finetuning, you structure training data in a consistent format. The **Alpaca format** (from Stanford's Alpaca project) has three sections:

1. **Instruction** — what the model should do → `"Given unoptimized LLVM IR, produce the equivalent O3-optimized IR."`
2. **Input** — the specific input → the unoptimized IR code
3. **Response** — the expected output → the O3-optimized IR code

**During training**, the model sees the complete template (instruction + input + response) and learns to predict the response.
**During inference**, you provide instruction + input with an **empty** response (`""`), and the model generates it.

### The `to_text` function

```python
def to_text(batch):
    texts.append(alpaca_prompt.format(instruction, inp, out) + EOS_TOKEN)
```

This formats each dataset example into the Alpaca template and appends the **EOS (End of Sequence) token**. The EOS token teaches the model **when to stop generating** — without it, the model would keep outputting text forever.

### The `.map()` call

```python
dataset = train_ds.map(to_text, batched=True, batch_size=1000, num_proc=8, ...)
```

- **`batched=True`** — processes data in batches instead of one-by-one (much faster)
- **`batch_size=1000`** — 1000 examples per batch (prevents RAM crash on large datasets)
- **`num_proc=8`** — uses 8 CPU cores in parallel for speed
- **`remove_columns=train_ds.column_names`** — drops the original columns, keeping only the new `"text"` column

---

## 4. Applying LoRA — The Heart of Efficient Finetuning (Lines 75–91)

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 256,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = 'unsloth'
)
```

> [!IMPORTANT]
> This is the most important section. This is where **QLoRA** is configured.

### What is LoRA? (Low-Rank Adaptation)

**The Problem**: A full finetune of an 8B model means updating all 8 billion parameters. This needs massive GPU memory (for weights + gradients + optimizer states).

**LoRA's Solution**: Instead of updating the original weight matrices, **freeze them entirely** and attach tiny trainable "adapter" matrices alongside them.

Here's the math. Say you have a weight matrix **W** of size `4096 × 4096` (= 16.7 million parameters). LoRA decomposes the update into two small matrices:

```
Original: W (4096 × 4096) → FROZEN, not trained
LoRA:     A (4096 × r) × B (r × 4096) → TRAINABLE
Output:   y = W·x + (A·B)·x
```

Where **r** (rank) is much smaller than 4096. With `r=128`:
- Original matrix: 4096 × 4096 = **16.7M parameters**
- LoRA matrices: 4096 × 128 + 128 × 4096 = **1.05M parameters** (6% of original!)

You only train the small A and B matrices, but the model behaves almost as if you finetuned the whole thing.

### What is QLoRA? (Quantized LoRA)

**QLoRA = Quantization + LoRA together**:

1. **The base model weights are loaded in 4-bit** (the `load_in_4bit=True` from earlier) — saves memory
2. **LoRA adapter matrices train in bfloat16/float16** — preserves learning quality
3. During forward pass: 4-bit base weights are **dequantized on-the-fly** to bfloat16, combined with LoRA adapters

This gives you the memory savings of both quantization AND LoRA simultaneously.

### What is PEFT?

**PEFT (Parameter-Efficient Fine-Tuning)** is the umbrella term for techniques like LoRA. The `get_peft_model()` function wraps the base model with LoRA adapters.

### Each Parameter Explained:

**`r = 128` (Rank)**
The "bottleneck" dimension of the LoRA matrices. Higher rank = more learnable parameters = better learning capacity but more memory. Typical values:

| Rank | Parameters per layer | Quality | Memory |
|---|---|---|---|
| 8 | Very few | Basic | Tiny |
| 32 | Moderate | Good | Small |
| 64 | More | Very good | Medium |
| **128** | **Lots** | **Excellent** | **Higher (A100/H100 can handle it)** |

Your rank of 128 is quite high — made possible by the A100/H100's 40GB+ VRAM.

**`target_modules`** — Which layers get LoRA adapters. These are the **attention** and **MLP** (feed-forward) layers inside each Transformer block:

| Module | Full Name | Part of |
|---|---|---|
| `q_proj` | Query projection | Attention |
| `k_proj` | Key projection | Attention |
| `v_proj` | Value projection | Attention |
| `o_proj` | Output projection | Attention |
| `gate_proj` | Gate projection | MLP (SwiGLU) |
| `up_proj` | Up projection | MLP |
| `down_proj` | Down projection | MLP |

By targeting all 7 modules, you're doing a **comprehensive** finetune. Some people only target `q_proj` and `v_proj` to save memory, but more modules = better quality.

### Quick Aside: What Are These Modules?

A Transformer model (like Qwen3) is made of many **Transformer blocks** stacked on top of each other. Each block has two main parts:

**1. Self-Attention** — lets the model look at all tokens and decide which ones are relevant:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I provide?"
- **O (Output)**: Projects the attention result back to the model dimension

**2. MLP (Feed-Forward Network)** — processes each token's representation:
- **gate_proj + up_proj**: SwiGLU activation (a modern activation function that works better than ReLU)
- **down_proj**: Projects back to model dimension

**`lora_alpha = 256`** (= 128 × 2)
A **scaling factor** for the LoRA update. The effective LoRA contribution is scaled by `alpha / r`.  With alpha=256 and r=128, the scaling = 2.0. A common rule of thumb is to set alpha = 2 × rank.

**`lora_dropout = 0.05`**
During training, randomly zeroes out 5% of LoRA connections. This is **regularization** — it prevents the model from overfitting (memorizing training data instead of learning patterns). 5% is mild — enough to help without hurting learning.

**`bias = "none"`**
Don't train bias terms in the LoRA layers. Biases are small and don't contribute much; leaving them frozen saves memory and prevents overfitting.

**`use_gradient_checkpointing = 'unsloth'`**

### Gradient Checkpointing

During training, the forward pass creates **activations** (intermediate results) at every layer. Normally, all activations are stored in memory for the backward pass (computing gradients). For a 8B model with thousands of layers, this uses a LOT of memory.

**Gradient checkpointing** trades **compute for memory**: it only stores activations at certain "checkpoints" and recomputes the rest during the backward pass. This roughly **halves memory usage** at the cost of ~30% slower training.

The `'unsloth'` variant is Unsloth's optimized version that's faster than the standard HuggingFace implementation.

---

## 5. Pre-Training Length Filter (Lines 95–107)

```python
MAX_SEQ_LENGTH = 8192
def is_within_length(batch):
    token_counts = [len(tokenizer.encode(t)) for t in batch["text"]]
    return [c <= MAX_SEQ_LENGTH for c in token_counts]
dataset = dataset.filter(is_within_length, batched=True, batch_size=64)
```

> [!WARNING]
> This filter is **critical for data quality**.

Without it, the SFTTrainer silently **truncates** examples longer than 8192 tokens. If the model trains on truncated IR, it learns that incomplete IR is valid output — which corrupts its understanding. This filter **removes** those examples entirely so training data stays clean.

---

## 6. Training Configuration (Lines 110–135)

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=8192,
    args=SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        warmup_steps=15,
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy='steps',
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        output_dir="/content/drive/MyDrive/mini-prot-checkpoints",
        optim='adamw_8bit',
    )
)
```

### SFTTrainer (Supervised Fine-Tuning Trainer)

From the `trl` library (Transformer Reinforcement Learning). **SFT** is the simplest form of finetuning:
- Show the model (input, expected_output) pairs
- The model learns to predict the expected output given the input
- Loss = how different the model's prediction is from the expected output (cross-entropy loss)

Unlike RLHF (Reinforcement Learning from Human Feedback), SFT doesn't need reward models or preference data. It just does supervised learning on your text pairs.

### Each Training Hyperparameter:

**`per_device_train_batch_size = 8`**
Process 8 examples **per GPU** in each forward pass. H100 has enough memory for this.

**`gradient_accumulation_steps = 8`**
Instead of updating model weights after every batch of 8, accumulate gradients over 8 batches, **then** update. This creates an **effective batch size of 8 × 8 = 64**.

Why? Larger effective batch sizes give more stable gradient estimates. But loading 64 examples at once might not fit in VRAM. Gradient accumulation simulates a large batch without needing the memory.

**`warmup_steps = 15`**
For the first 15 training steps, the learning rate **gradually increases** from 0 to the target (2e-4). This prevents the model from making wild, destructive weight updates at the beginning when gradients are noisy.

**`num_train_epochs = 1`**
One pass through the entire training dataset. More epochs = more learning, but risk overfitting. For 19,800 samples with effective batch size 64, that's roughly 19800/64 ≈ **309 training steps**.

**`learning_rate = 2e-4`** (0.0002)
How big each weight update is. Too high = training becomes unstable (model "overshoots"). Too low = training is too slow. 2e-4 is standard for LoRA finetuning.

**`lr_scheduler_type = "cosine"`**

### Cosine Learning Rate Schedule

Instead of a constant learning rate, it follows a cosine curve:

```
LR
↑
2e-4 |╲
     |  ╲
     |    ╲
     |      ╲_____
0    |____________→ Training Steps
     warmup  decay
```

After warmup, the LR starts high and slowly decays following a cosine curve. This lets the model make big updates early (to learn fast) and small updates later (to fine-tune).

**`weight_decay = 0.01`**
A form of **regularization**. Slightly penalizes large weights, pushing them toward zero. Prevents overfitting by discouraging the model from relying on any single weight too much. 0.01 is a mild, standard value.

**`logging_steps = 10`**
Print training loss every 10 steps.

**`save_strategy='steps'` + `save_steps=50`**
Save a checkpoint every 50 steps. With ~309 total steps, you get ~6 checkpoints.

**`save_total_limit = 2`**
Only keep the 2 most recent checkpoints on disk (to save storage).

**`fp16=False, bf16=True`**
Use bfloat16 for training computations (matches the H100's native support, as discussed earlier).

**`optim = 'adamw_8bit'`**

### AdamW 8-bit Optimizer

**AdamW** is the go-to optimizer for training Transformers. It maintains two extra values (momentum + variance) for each parameter, which normally doubles memory usage.

**8-bit AdamW** quantizes these optimizer states to 8 bits, roughly halving the optimizer's memory footprint. This is crucial for fitting everything on one GPU. The quality loss is negligible.

---

## 7. Saving the LoRA Model (Lines 138–139)

```python
model.save_pretrained("/content/drive/MyDrive/mini-prot-lora-model")
tokenizer.save_pretrained("/content/drive/MyDrive/mini-prot-lora-model")
```

> [!NOTE]
> This only saves the **LoRA adapter weights** (the small A and B matrices), NOT the full 8B model. The saved files are typically only **100–500 MB** instead of ~16 GB for the full model. When loading later, you load the base model + LoRA adapters to reconstruct the finetuned model.

> [!CAUTION]
> The file is missing `trainer.train()` — the actual training call! It saves the model without training it. This line should exist between the trainer creation and `model.save_pretrained()`.

---
---

# Part 2: Evaluation — [mini_prot_eval_code.py](mini_prot_eval_code.py)

---

## 1. Loading the Finetuned Model (Lines 35–42)

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/mini-prot-lora-model",
    max_seq_length=8192,
    dtype=torch.float16,   # Note: float16 instead of bfloat16
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

Key differences from training:
- **Loads from your saved LoRA model path** — Unsloth automatically loads the base model + merges the LoRA adapters
- **`float16`** instead of `bfloat16` — this code was written for a T4 GPU (evaluation can run on cheaper hardware). T4s support float16 natively but NOT bfloat16
- **`for_inference(model)`** — enables optimizations for generation: disables dropout, enables 2x faster inference via Unsloth's custom kernels, turns off gradient computation

---

## 2. Reproducing the Same Test Split (Lines 45–58)

```python
ds = load_dataset("YangziResearch/IR-OptSet", split="train")
selected = ds.select(range(20000))
train_test = selected.train_test_split(test_size=200, seed=42)
test_ds = train_test["test"]
```

Uses the **exact same split logic** with the same `seed=42` to ensure the 200 test samples are **identical** to the ones excluded from training. This is crucial — if test data overlapped with training data, your evaluation would be meaningless (the model would just be regurgitating memorized examples).

---

## 3. The MCA Sanity Check: Measuring Optimization Quality

This is the evaluation metric — it measures how many **CPU cycles** the model's optimized IR would take versus the original LLVM O3 optimization.

### What is LLVM IR?

**LLVM IR (Intermediate Representation)** is a low-level programming language used internally by the LLVM compiler. When you compile C/C++ code, the compiler first translates it to IR, then optimizes it, then produces machine code. Think of it as an "assembly language for compilers."

### What is O3 Optimization?

LLVM has optimization levels `-O0` (none) through `-O3` (maximum). O3 applies dozens of optimization passes (dead code elimination, loop unrolling, inlining, vectorization, etc.) to make the code run faster. Your model is trained to replicate these optimizations.

### What is LLVM MCA?

**LLVM MCA (Machine Code Analyzer)** is a tool that **simulates** running assembly code on a specific CPU (in your case, Skylake) and reports how many **CPU cycles** it would take. Lower cycles = faster code = better optimization.

### The Pipeline for Each Test Sample:

```
┌─────────────────────┐
│   Unoptimized IR    │
│   (preprocessed)    │
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌──────────────────────┐
│ llc -O0│  │  Feed to LLM model   │
│  + MCA │  │  (generate output)   │
└───┬────┘  └──────────┬───────────┘
    │                  │
    │                  ▼
    │            ┌──────────┐
    │            │ llc -O0  │
    │            │  + MCA   │
    │            └────┬─────┘
    │                 │
    ▼                 ▼
 PRE Cycles      LLM Cycles
    │                 │      ┌──────────────┐
    │                 │      │ Ground Truth │
    │                 │      │  O3 IR       │
    │                 │      └──────┬───────┘
    │                 │             │
    │                 │             ▼
    │                 │       ┌──────────┐
    │                 │       │ llc -O0  │
    │                 │       │  + MCA   │
    │                 │       └────┬─────┘
    │                 │            │
    ▼                 ▼            ▼
┌─────────────────────────────────────────┐
│           COMPARE ALL THREE             │
│  PRE Cycles vs LLVM Cycles vs LLM Cycles│
└─────────────────────────────────────────┘
```

Three IR files are measured:
1. **PRE** — the original unoptimized IR (baseline)
2. **LLVM-O3** — the ground truth O3-optimized IR (what LLVM's compiler produces)
3. **LLM-O3** — what your finetuned model produces (the thing being evaluated)

---

## 4. The Heuristic Cycle Estimator (Lines 74–135)

```python
CYCLE_TABLE = {
    'add': 1, 'sub': 1, 'mul': 3, 'udiv': 25, ...
    'load': 4, 'store': 3, 'call': 10, ...
}

def heuristic_cycles(filepath):
    # Count IR opcodes and estimate total cycles
```

### Why a Heuristic?

When the model produces IR that has syntax errors (very common — look at your evaluation results, almost every row shows `Syntax_Error` for LLM-O3), the LLVM tools (`llc`, `llvm-mca`) fail. But you still want **some** estimate of quality.

The heuristic:
1. Reads the IR file line by line
2. Identifies each **opcode** (instruction like `add`, `load`, `mul`)
3. Looks up an estimated cycle cost from a hardcoded table (based on Skylake CPU timings)
4. Sums them up for a rough total

### The Loop Adjustment (Lines 131–133)

```python
if phi_count / bb_count > 0.3:
    total = int(total * (1 - 0.4) + total * 0.4 * 8)
```

**Phi nodes** (`phi` instructions) indicate loops in IR (they select values based on which basic block execution came from). If >30% of basic blocks have phi nodes, it means there are many loops. The heuristic multiplies 40% of the cycle count by 8 (assuming each loop iterates ~8 times on average) for a rough estimate.

---

## 5. The `get_mca_cycles` Function (Lines 138–189)

```python
def get_mca_cycles(filepath, skip_verify=True):
```

This is the pipeline that tries to get **real** cycle counts:

**Step 1: Optional Verification** (`skip_verify=False` mode)
```python
subprocess.run(["opt", "--passes=verify", "-disable-output", filepath])
```
Runs LLVM's verification pass to check if the IR is syntactically valid. If it fails → `Syntax_Error`.

The `skip_verify=True` default skips this and goes straight to `llc`, using `llc`'s own error handling for syntax validation instead.

**Step 2: Compile to Assembly**
```python
subprocess.run(["llc", "-O0", filepath, "-o", asm])
```
`llc` = LLVM static compiler. Converts IR → machine assembly. `-O0` means no optimizations during this step (you want to measure the IR's quality, not let llc optimize further).

**Step 3: Clean the Assembly**
```python
for line in f_in:
    if not line.strip().startswith('.'):
        f_out.write(line)
```
Removes **assembler directives** (lines starting with `.`) like `.text`, `.globl`, etc. LLVM MCA only needs actual instructions, not metadata.

**Step 4: Run LLVM MCA**
```python
subprocess.run(["llvm-mca", "-mcpu=skylake", clean_asm])
m = re.search(r"Total Cycles:\s+(\d+)", mca.stdout)
```
Simulates execution on a Skylake CPU and extracts the total cycle count from the output.

---

## 6. The `evaluate` and `pct` Functions (Lines 220–252)

```python
def pct(base, other):
    return f"{((base - other) / max(base, 1)) * 100:.2f}%"
```

Calculates **percentage improvement**: `(base - other) / base × 100`. Positive = `other` is faster. Negative = `other` is slower.

```python
def evaluate(pre_ir_path, llvm_ir_path, llm_ir_path):
    pre = get_mca_cycles(pre_ir_path)
    llvm = get_mca_cycles(llvm_ir_path)
    llm = get_mca_cycles(llm_ir_path)
```

Gets cycle counts for all three and computes comparisons. If MCA fails for the LLM output (syntax error), it falls back to the **heuristic estimator** and marks the result with `(est)`.

---

## 7. The Main Evaluation Loop (Lines 271–318)

```python
for i, example in enumerate(test_ds):
    # 1. Get the input (unoptimized IR) and ground truth (O3 IR)
    pre_ir = example["preprocessed_ir"]
    llvm_ir = example["o3_ir"]

    # 2. Create the prompt with empty response
    prompt = alpaca_prompt.format(
        "Given unoptimized LLVM IR, produce the equivalent O3-optimized IR.",
        pre_ir, "",  # Empty response — model must generate it
    )

    # 3. Tokenize and run inference
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=8192,
                             temperature=0.3, do_sample=True, top_p=0.9)

    # 4. Extract model's response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    llm_ir = response.split("### Response:\n")[-1]

    # 5. Write all three IRs to files and measure cycles
    # 6. Log to CSV
```

### Generation Parameters:

**`max_new_tokens=8192`** — generate up to 8192 new tokens (the response can be as long as the input)

**`temperature=0.3`** — controls randomness. Lower = more deterministic/conservative. 0.3 is fairly low, meaning the model picks high-confidence tokens:

| Temperature | Behavior |
|---|---|
| 0.0 | Always picks the most likely token (greedy) |
| **0.3** | **Mostly picks likely tokens, slight variety** |
| 1.0 | Standard sampling |
| 2.0 | Very random, creative |

**`do_sample=True`** — enables sampling (random selection from the probability distribution). With `False`, it would be greedy decoding (always picking the #1 most likely token).

**`top_p=0.9`** — **Nucleus sampling**. Only considers tokens whose cumulative probability adds up to 90%. This filters out the long tail of unlikely tokens while preserving natural variation. For example, if the top 5 tokens cover 90% probability, only those 5 are considered.

**`use_cache=True`** — enables **KV-cache** (Key-Value cache). During autoregressive generation, the model generates one token at a time. Without cache, it recomputes attention for ALL previous tokens every step. With cache, it stores the Key and Value matrices and only computes the new token — dramatically faster.

**Token length check (Line 282)**:
```python
if inputs['input_ids'].shape[1] > 24000:
    print(f'Skipping huge {inputs["input_ids"].shape[1]} token file...')
```
Skips inputs that are too large (>24K tokens) to avoid OOM (Out of Memory) errors on the GPU.

---

## 8. Reading the Evaluation Results

From the output at the bottom of the file, here's what the columns mean:

| Column | Meaning |
|---|---|
| `PRE` | CPU cycles for unoptimized IR |
| `LLVM-O3` | Cycles for LLVM's O3 optimization (ground truth) |
| `LLM-O3` | Cycles for model's output (what you're measuring) |
| `Pre vs LLVM` | How much faster LLVM O3 is vs unoptimized (positive = LLVM is better) |
| `Pre vs LLM` | How much faster model's output is vs unoptimized |
| `LLVM vs LLM` | How model compares to LLVM O3 (negative = model is worse) |
| `Est LLM vs PRE` | Estimated improvement (when MCA fails, uses heuristic) |
| `Est LLM vs LLVM` | Estimated comparison to LLVM (heuristic) |

### What the Results Show

Looking at your sample output, almost every LLM-O3 result is `Syntax_Error`. This means the model is generating IR that isn't syntactically valid — the LLVM tools can't compile it. Row 176 is an exception where the model produced valid IR with 3056 cycles vs LLVM's 3007 cycles (the model was 1.63% worse — close to LLVM quality!).

> [!NOTE]
> Syntax errors are a common problem with LLM-generated code, especially structured code like LLVM IR. This suggests the model may need more training data, more epochs, or a different approach (like constrained decoding) to improve syntactic correctness.

---

## Summary: The Full QLoRA Finetuning Pipeline

```
┌───────────────────────────┐     ┌──────────────────────────┐
│  Base Model               │     │  IR-OptSet Dataset       │
│  Qwen3-8B (8B params)     │     │  20,000 examples         │
└─────────────┬─────────────┘     └────────────┬─────────────┘
              │                                 │
              │ load_in_4bit=True                │ Alpaca format
              │ (NF4 Quantization)               │ + length filter
              ▼                                 ▼
┌───────────────────────────┐     ┌──────────────────────────┐
│  4-bit Quantized Model    │     │  Formatted Training Data │
│  (~5 GB in VRAM)          │     │  (text column)           │
└─────────────┬─────────────┘     └────────────┬─────────────┘
              │                                 │
              │ get_peft_model()                 │
              │ (LoRA adapters added)            │
              ▼                                 │
┌───────────────────────────┐                   │
│  QLoRA Model              │                   │
│  Frozen 4-bit base        │                   │
│  + Trainable LoRA (r=128) │                   │
└─────────────┬─────────────┘                   │
              │                                 │
              └──────────────┬──────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │  SFTTrainer              │
              │  (Supervised Fine-Tuning)│
              │  AdamW 8-bit, cosine LR  │
              │  gradient checkpointing  │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  Trained LoRA Adapters   │
              │  (~100-500 MB saved)     │
              └────────────┬─────────────┘
                           │
                           │ Load on eval GPU
                           ▼
              ┌──────────────────────────┐
              │  Inference               │
              │  Generate optimized IR   │
              └────────────┬─────────────┘
                           │
                           │ llc + llvm-mca
                           ▼
              ┌──────────────────────────┐
              │  Cycle Count Evaluation  │
              └──────────────────────────┘
```

| Technique | Purpose | Memory Savings |
|---|---|---|
| **4-bit Quantization (NF4)** | Compress base model weights | ~75% less |
| **LoRA (r=128)** | Only train small adapter matrices | Train <2% of params |
| **Gradient Checkpointing** | Recompute activations instead of storing | ~50% less |
| **AdamW 8-bit** | Quantize optimizer states | ~50% less on optimizer |
| **bfloat16 Training** | Half-precision math | ~50% less on activations |

Together, these make it possible to finetune an **8 billion parameter** model on a single GPU that would otherwise need an entire server cluster.

from google.colab import drive
drive.mount('/content/drive')


import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from unsloth import FastLanguageModel
import json
from datasets import Dataset# Load the model
import torch
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B-Base",
    max_seq_length = 8192,   # longer context to accommodate IR code blocks
    dtype = torch.bfloat16,   # A100/H100 natively supports bfloat16 for better precision
    load_in_4bit = True,
)

# 2️⃣ Load dataset from HuggingFace
from datasets import load_dataset

ds = load_dataset("YangziResearch/IR-OptSet", split="train")

# --- A100 CONFIGURATION ---
DRY_RUN = False  # Set to False to train on the full 170k dataset!


# Split the dataset based on DRY_RUN mode
if DRY_RUN:
    print("Running in DRY_RUN mode (2200 samples total)...")
    ds = ds.select(range(2200))
    train_test = ds.train_test_split(test_size=200, seed=42)
else:
    print("Running in FULL mode (10k samples)...")
    selected = ds.select(range(20000))
    train_test = selected.train_test_split(test_size=200, seed=42)

train_ds = train_test["train"]
test_ds = train_test["test"]

# 3️⃣ Alpaca-style prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def to_text(batch):
    instruction = "Given unoptimized LLVM IR, produce the equivalent O3-optimized IR."
    texts = []
    for inp, out in zip(batch['preprocessed_ir'], batch['o3_ir']):
        if not isinstance(inp, str): inp = json.dumps(inp, ensure_ascii=False)
        if not isinstance(out, str): out = json.dumps(out, ensure_ascii=False)
        texts.append(alpaca_prompt.format(instruction, inp, out) + EOS_TOKEN)
    return {"text": texts}



# CRITICAL FIX for 170k dataset: batched processing to prevent RAM crash
dataset = train_ds.map(
    to_text,
    batched=True,
    batch_size=1000,
    num_proc=8,
    remove_columns=train_ds.column_names
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128,  # A100/H100 has 40GB+ — can handle higher rank for better learning
    target_modules= [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_alpha = 128 * 2,  # keep 2x rank
    lora_dropout = 0.05,  # small dropout to prevent overfitting on 1k samples
    bias = "none",
    use_gradient_checkpointing='unsloth'
)



# ── Pre-training length filter ──────────────────────────────────────────────
# SFTTrainer silently truncates examples that exceed max_seq_length.
# Training on truncated (cut-off) IR teaches the model that incomplete IR is
# valid output. We filter them out entirely to keep the training data clean.
MAX_SEQ_LENGTH = 8192
def is_within_length(batch):
    token_counts = [len(tokenizer.encode(t)) for t in batch["text"]]
    return [c <= MAX_SEQ_LENGTH for c in token_counts]
before_count = len(dataset)
dataset = dataset.filter(is_within_length, batched=True, batch_size=64)
after_count = len(dataset)
print(f"Dataset filtered: {before_count} → {after_count} examples "
      f"({before_count - after_count} removed due to length > {MAX_SEQ_LENGTH} tokens)")
# ─────────────────────────────────────────────────────────────────────────────

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=8192,
    args=SFTConfig(
        per_device_train_batch_size=8,      # H100 natively handles large batches
        gradient_accumulation_steps=8,      # effective batch = 64
        warmup_steps=10 if DRY_RUN else 15,                    # ~10% of 1 epoch (~140 steps for 10k run)
        num_train_epochs=1,                 # 1 pass over the 9k training data
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=1 if DRY_RUN else 10,
        save_strategy='steps',
        save_steps=50,                      # Save more frequently since total steps is ~140
        save_total_limit=2,
        fp16=False,                          # H100 strongly prefers bf16
        bf16=True,                           # H100 natively supports bf16
        output_dir="/content/drive/MyDrive/mini-prot-checkpoints",
        optim='adamw_8bit',
    )
)


model.save_pretrained("/content/drive/MyDrive/mini-prot-lora-model")
tokenizer.save_pretrained("/content/drive/MyDrive/mini-prot-lora-model")
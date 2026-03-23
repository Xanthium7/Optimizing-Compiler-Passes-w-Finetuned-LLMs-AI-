# 🚀 Optimizing Compiler Passes with Fine-Tuned LLMs

<div align="center">

**Teaching Large Language Models to perform LLVM O3-level compiler optimizations on Intermediate Representation (IR) code**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![LLVM](https://img.shields.io/badge/LLVM-18-262D3A?logo=llvm&logoColor=white)](https://llvm.org)
[![Unsloth](https://img.shields.io/badge/Unsloth-LoRA-FF6F00)](https://github.com/unslothai/unsloth)
[![HuggingFace](https://img.shields.io/badge/🤗_Dataset-IR--OptSet-FFD21E)](https://huggingface.co/datasets/YangziResearch/IR-OptSet)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Details](#-model-details)
- [Evaluation Pipeline](#-evaluation-pipeline)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Results](#-results)
- [Contributing](#-contributing)

---

## 🔍 Overview

Traditional compiler optimizations rely on hand-crafted heuristics and rule-based transformations that have been refined over decades. This project explores a fundamentally different approach: **fine-tuning a Large Language Model to learn compiler optimization passes directly from data**.

Given unoptimized LLVM IR (equivalent to `-O0`), the model predicts optimized IR that mirrors the transformations typically applied by LLVM's `-O3` optimization pipeline — including dead code elimination, constant propagation, loop optimizations, register allocation improvements, and more.

### Key Highlights

- 🧠 **Fine-tuned Qwen3-8B** model using QLoRA (4-bit quantization) for efficient training
- 📊 **20,000 IR pairs** from the [IR-OptSet](https://huggingface.co/datasets/YangziResearch/IR-OptSet) dataset
- ⚡ **Hardware-aware evaluation** using `llvm-mca` cycle estimation on Skylake microarchitecture
- 📈 **Heuristic fallback** estimator for cases where compilation fails
- 🔬 **Three-way comparison**: Unoptimized (O0) vs. LLVM O3 vs. LLM-predicted IR

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                        │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │  IR-OptSet   │───▷│  Prompt       │───▷│  Fine-Tune         │  │
│  │  Dataset     │    │  Formatting   │    │  Qwen3-8B (QLoRA) │  │
│  │  (20K pairs) │    │              │    │  LoRA r=128        │  │
│  └─────────────┘    └──────────────┘    └────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                      Evaluation Pipeline                         │
│                                                                  │
│  ┌───────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐ │
│  │ Pre-Opt   │    │ LLVM O3   │    │ LLM      │    │ Compare  │ │
│  │ IR (O0)   │───▷│ IR        │───▷│ Predicted│───▷│ Cycles   │ │
│  │           │    │ (Ground   │    │ IR       │    │ (MCA)    │ │
│  │           │    │  Truth)   │    │          │    │          │ │
│  └───────────┘    └───────────┘    └──────────┘    └──────────┘ │
│                                          │                       │
│                                    ┌─────▼──────┐               │
│                                    │ Heuristic   │               │
│                                    │ Fallback    │               │
│                                    │ (if compile │               │
│                                    │  fails)     │               │
│                                    └────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
.
├── mini_prot_finetune_H100_Ready.ipynb   # Fine-tuning notebook (Colab/H100)
├── mini_prot_evaluation_only.ipynb       # Evaluation pipeline notebook
├── mca_sanity_check.py                   # llvm-mca cycle estimation & comparison
├── LLM-O3 Performance Evaluation Results FINAL.xlsx  # Evaluation results spreadsheet
├── raw.ll                                # Example: raw Clang IR output
├── O0.ll                                 # Example: unoptimized IR (O0)
├── O3.ll                                 # Example: LLVM O3-optimized IR
├── testing-preIR.txt                     # Sample unoptimized IR for testing
├── testing-optIR.txt                     # Sample O3-optimized IR for testing
└── README.md                             # This file
```

---

## 📊 Dataset

This project uses the [**YangziResearch/IR-OptSet**](https://huggingface.co/datasets/YangziResearch/IR-OptSet) dataset from HuggingFace, which contains pairs of:

| Field | Description |
|-------|-------------|
| **Pre-optimization IR** | Unoptimized LLVM IR generated by Clang at `-O0` |
| **Post-optimization IR** | Corresponding LLVM IR after applying `-O3` passes |

### Data Split

| Split | Size | Purpose |
|-------|------|---------|
| Training | 20,000 samples | Fine-tuning the model |
| Test | 200 samples | Evaluation & benchmarking |

The IR code targets the `x86_64` architecture and is compiled from C/C++ source files using Clang.

---

## 🧠 Model Details

### Base Model

| Parameter | Value |
|-----------|-------|
| **Model** | [`unsloth/Qwen3-8B-Base`](https://huggingface.co/unsloth/Qwen3-8B-Base) |
| **Parameters** | 8 Billion |
| **Quantization** | 4-bit (QLoRA via `bitsandbytes`) |
| **Max Sequence Length** | 8,192 tokens |
| **Precision** | `bfloat16` (for A100/H100 GPUs) |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| **Rank (r)** | 128 |
| **Alpha** | 128 (implied 1:1 ratio) |
| **Target Modules** | All linear layers |
| **Dropout** | 0 |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Epochs** | 1 |
| **Learning Rate** | 2e-4 |
| **Batch Size** | 8 per device |
| **Optimizer** | AdamW (8-bit) |
| **Scheduler** | Linear decay |
| **Framework** | [Unsloth](https://github.com/unslothai/unsloth) + HuggingFace TRL |

---

## 📏 Evaluation Pipeline

The evaluation system measures how well the LLM-predicted IR performs compared to both the unoptimized baseline and LLVM's own O3 output.

### Cycle Estimation Methods

#### 1. `llvm-mca` (Primary)

The gold-standard approach uses LLVM's Machine Code Analyzer:

```
IR → llc (compile to assembly) → llvm-mca -mcpu=skylake → Total Cycles
```

- Targets **Intel Skylake** microarchitecture
- Strips assembler directives before analysis
- Provides precise per-instruction throughput estimates

#### 2. Heuristic Estimator (Fallback)

When LLM-generated IR fails to compile (syntax errors), a heuristic cycle estimator kicks in:

```python
# Skylake cycle costs per LLVM IR opcode
CYCLE_TABLE = {
    'add': 1, 'sub': 1, 'mul': 3,
    'udiv': 25, 'sdiv': 25,
    'load': 4, 'store': 3,
    'call': 10, 'fadd': 4, 'fdiv': 14,
    ...
}
```

- Parses IR text and sums estimated cycle costs per opcode
- Applies loop-scaling heuristic when phi-node density suggests loops
- Results are marked with `(est)` suffix

### Metrics

For each test sample, the pipeline computes:

| Metric | Description |
|--------|-------------|
| **Pre-opt Cycles** | Cycle count of unoptimized (O0) IR |
| **LLVM O3 Cycles** | Cycle count of LLVM-optimized IR |
| **LLM Cycles** | Cycle count of LLM-predicted IR |
| **Pre vs LLVM (%)** | Speedup of LLVM O3 over baseline |
| **Pre vs LLM (%)** | Speedup of LLM output over baseline |
| **LLVM vs LLM (%)** | How close LLM gets to LLVM O3 performance |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.10+
- **LLVM 18** (`llc`, `opt`, `llvm-mca` must be on PATH)
- **GPU** with ≥24 GB VRAM (H100/A100 recommended for training)
- **Google Colab Pro+** (recommended for notebook execution)

### Installation

```bash
# Clone the repository
git clone https://github.com/Xanthium7/Optimizing-Compiler-Passes-w-Finetuned-LLMs-AI-.git
cd Optimizing-Compiler-Passes-w-Finetuned-LLMs-AI-

# Install LLVM 18 (Ubuntu/Debian)
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18

# Install Python dependencies
pip install unsloth transformers datasets trl bitsandbytes accelerate

# Verify LLVM tools are available
llc --version
opt --version
llvm-mca --version
```

---

## 💻 Usage

### 1. Fine-Tuning (Google Colab)

Open [`mini_prot_finetune_H100_Ready.ipynb`](mini_prot_finetune_H100_Ready.ipynb) in Google Colab with a **H100 GPU runtime** and run all cells. The notebook will:

1. Install Unsloth and dependencies
2. Load the Qwen3-8B base model in 4-bit precision
3. Attach LoRA adapters (r=128)
4. Download and format the IR-OptSet dataset
5. Fine-tune for 1 epoch
6. Save the LoRA adapter weights to Google Drive

### 2. Evaluation (Google Colab)

Open [`mini_prot_evaluation_only.ipynb`](mini_prot_evaluation_only.ipynb) and run all cells. The notebook will:

1. Install LLVM 18 on the Colab instance
2. Load the fine-tuned model (LoRA adapters from Drive)
3. Run inference on the test split
4. Evaluate each sample using `llvm-mca` or the heuristic fallback
5. Log results to `mca_evaluation_log.csv`

### 3. Local Sanity Check

Use the standalone evaluation script for quick local testing:

```bash
# Two-file comparison (pre-optimization vs optimized)
python mca_sanity_check.py raw.ll O3.ll

# Three-file evaluation (pre-opt vs LLVM O3 vs LLM output)
python mca_sanity_check.py pre.ll llvm_o3.ll llm_predicted.ll
```

**Example output:**
```
============================================================
LLVM Code Analysis
============================================================

PRE IR  (raw.ll):
  Result: 42

O3 IR   (O3.ll) [skipping opt --passes=verify]:
  Result: 18

============================================================
SUCCESS: O3 is faster by 24 cycles (57.14% reduction)
============================================================
```

---

## 📈 Results

Detailed evaluation results are available in [`LLM-O3 Performance Evaluation Results FINAL.xlsx`](LLM-O3%20Performance%20Evaluation%20Results%20FINAL.xlsx).

The evaluation measures how closely the LLM's predicted optimizations approach LLVM's O3 pass pipeline in terms of estimated CPU cycles on a Skylake processor.

---

## 🛠 Technical Details

### Why `skip_verify=True` for O3 IR?

LLVM O3-optimized IR may use newer IR constructs or intrinsics that fail LLVM's strict `opt --passes=verify` check, especially across LLVM versions. The evaluation pipeline uses `skip_verify=True` for O3 ground-truth IR and relies on `llc` for syntax validation instead. This prevents false negatives where valid O3 output is flagged as erroneous due to version mismatches.

### Heuristic Loop Scaling

When the ratio of `phi` nodes to basic blocks exceeds 30%, the heuristic estimator applies a loop-scaling factor:

```
total = int(total * 0.6 + total * 0.4 * 8)
```

This approximates the effect of loop iterations, providing a more realistic cycle estimate for loop-heavy code.

---

## 🤝 Contributing

Contributions are welcome! Here are some areas for improvement:

- 🔧 Support for additional target architectures (ARM, RISC-V)
- 📊 Additional evaluation metrics (code size, memory usage)
- 🧪 Larger training datasets and longer training schedules
- 🏗 Support for function-level and module-level optimizations
- 📈 Comparison with other LLM architectures

---

## 📄 License

This project is open source. See the repository for license details.

---

<div align="center">

**Built with ❤️ using [Unsloth](https://github.com/unslothai/unsloth), [LLVM](https://llvm.org), and [HuggingFace](https://huggingface.co)**

</div>

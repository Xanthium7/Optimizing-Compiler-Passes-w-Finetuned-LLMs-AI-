'''
from google.colab import drive
drive.mount('/content/drive')


!apt-get update && apt-get install -y llvm

%pip install unsloth

!wget https://apt.llvm.org/llvm.sh
!chmod +x llvm.sh
!sudo ./llvm.sh 18
!apt-get install -y llvm-18
!update-alternatives --install /usr/bin/opt opt /usr/lib/llvm-18/bin/opt 100
!update-alternatives --install /usr/bin/llc llc /usr/lib/llvm-18/bin/llc 100
!update-alternatives --install /usr/bin/llvm-mca llvm-mca /usr/lib/llvm-18/bin/llvm-mca 100



'''


import logging
import warnings
import os
import sys
import re
import subprocess
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch

# For T4 GPUs, we MUST use float16 instead of bfloat16
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/content/drive/MyDrive/mini-prot-lora-model",  # Path to saved LoRA
    max_seq_length=8192,
    dtype=torch.float16,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)  # Enable native 2x faster inference


ds = load_dataset("YangziResearch/IR-OptSet", split="train")

DRY_RUN = False  # Set to False for full evaluation

if DRY_RUN:
    print("Running in DRY_RUN mode (200 test samples)...")
    ds = ds.select(range(2200))
    train_test = ds.train_test_split(test_size=200, seed=42)
else:
    print("Running in FULL EVALUATION mode (200 test samples from 20k)...")
    selected = ds.select(range(20000))
    train_test = selected.train_test_split(test_size=200, seed=42)

test_ds = train_test["test"]

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


% % writefile mca_sanity_check.py

# Skylake cycle costs per LLVM IR opcode
CYCLE_TABLE = {
    'add': 1, 'sub': 1, 'and': 1, 'or': 1, 'xor': 1, 'shl': 1, 'lshr': 1, 'ashr': 1,
    'icmp': 1, 'select': 1, 'zext': 1, 'sext': 1, 'trunc': 1, 'bitcast': 0,
    'phi': 0, 'alloca': 1, 'extractvalue': 1, 'insertvalue': 1,
    'getelementptr': 1,
    'mul': 3,
    'udiv': 25, 'sdiv': 25, 'urem': 25, 'srem': 25,
    'fadd': 4, 'fsub': 4, 'fmul': 4, 'frem': 20,
    'fdiv': 14,
    'fptoui': 3, 'fptosi': 3, 'uitofp': 3, 'sitofp': 3, 'fpext': 1, 'fptrunc': 1,
    'load': 4, 'store': 3, 'atomicrmw': 10, 'cmpxchg': 15,
    'br': 1, 'ret': 1, 'switch': 2, 'indirectbr': 3, 'unreachable': 0,
    'call': 10, 'invoke': 10, 'tail': 10,
}


def heuristic_cycles(filepath):
    try:
        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return None

    opcode_re = re.compile(
        r'^\s+(?:%[\w.]+\s*=\s*)?'
        r'(tail\s+)?'
        r'([\w.]+)'
    )
    total = 0
    phi_count = 0
    bb_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';') or stripped.startswith('!') \
                or stripped.startswith('source_filename') \
                or stripped.startswith('target') \
                or stripped.startswith('attributes') \
                or stripped.startswith('define') \
                or stripped.startswith('declare') \
                or stripped.startswith('}'):
            continue

        if re.match(r'^[\w.][\w.]*:', stripped):
            bb_count += 1
            continue

        m = opcode_re.match(line)
        if not m:
            continue
        opcode = m.group(2).lower()
        cost = CYCLE_TABLE.get(opcode, 0)
        total += cost
        if opcode == 'phi':
            phi_count += 1

    if bb_count > 0 and phi_count / bb_count > 0.3:
        loop_fraction = 0.4
        total = int(total * (1 - loop_fraction) + total * loop_fraction * 8)

    return total if total > 0 else None


def get_mca_cycles(filepath, skip_verify=True):
    """
    Try to get cpu cycles for the given IR file.

    If skip_verify=False, runs opt --passes=verify first (strict mode).
    If skip_verify=True, skips opt and goes straight to llc.
    """
    if not skip_verify:
        chk = subprocess.run(
            ["opt", "--passes=verify", "-disable-output", filepath],
            capture_output=True
        )
        if chk.returncode != 0:
            err = chk.stderr.decode()[:120].strip().replace(',', ';')
            return f"Syntax_Error: {err}"

    asm = filepath + ".s"
    clean_asm = filepath + "_clean.s"
    try:
        # Step 1: Compile to O0 assembly (no further optimizations)
        result = subprocess.run(
            ["llc", "-O0", filepath, "-o", asm],
            capture_output=True
        )
        if result.returncode != 0:
            # llc itself failed — output the error safely so the table parses it
            err = result.stderr.decode()[:120].strip().replace(',', ';')
            return f"Syntax_Error: LLC_Failed: {err}"

        # Step 2: Clean assembly of directives
        with open(asm, 'r') as f_in, open(clean_asm, 'w') as f_out:
            for line in f_in:
                if not line.strip().startswith('.'):
                    f_out.write(line)

        # Step 3: Run LLVM MCA
        mca = subprocess.run(
            ["llvm-mca", "-mcpu=skylake", clean_asm],
            capture_output=True, text=True, check=True
        )
        m = re.search(r"Total Cycles:\s+(\d+)", mca.stdout)

        # Cleanup
        if os.path.exists(asm):
            os.remove(asm)
        if os.path.exists(clean_asm):
            os.remove(clean_asm)

        return int(m.group(1)) if m else "MCA_Failed"

    except Exception as e:
        return "MCA_Error"


# def get_mca_cycles(filepath):
#     chk = subprocess.run(["opt", "--passes=verify", "-disable-output", filepath],
#                          capture_output=True)
#     if chk.returncode != 0:
#         err = chk.stderr.decode()[:120].strip().replace(',', ';')
#         return f"Syntax_Error: {err}"

#     asm = filepath + ".s"
#     clean_asm = filepath + "_clean.s"
#     try:
#         subprocess.run(["llc", "-O0", filepath, "-o", asm],
#                        check=True, capture_output=True)
#         with open(asm, 'r') as f_in, open(clean_asm, 'w') as f_out:
#             for line in f_in:
#                 if not line.strip().startswith('.'):
#                     f_out.write(line)
#         result = subprocess.run(["llvm-mca", "-mcpu=skylake", clean_asm],
#                                 capture_output=True, text=True, check=True)
#         m = re.search(r"Total Cycles:\s+(\d+)", result.stdout)
#         if os.path.exists(asm):
#             os.remove(asm)
#         if os.path.exists(clean_asm):
#             os.remove(clean_asm)
#         return int(m.group(1)) if m else "MCA_Failed"
#     except Exception:
#         return "MCA_Error"


def pct(base, other):
    if not isinstance(base, int) or not isinstance(other, int):
        return "N/A"
    return f"{((base - other) / max(base, 1)) * 100:.2f}%"


def evaluate(pre_ir_path, llvm_ir_path, llm_ir_path):
    pre = get_mca_cycles(pre_ir_path)
    llvm = get_mca_cycles(llvm_ir_path)
    llm = get_mca_cycles(llm_ir_path)

    real_pre_vs_llm = pct(pre,  llm)
    real_llvm_vs_llm = pct(llvm, llm)

    if isinstance(llm, int):
        est_pre = real_pre_vs_llm
        est_llvm = real_llvm_vs_llm
    else:
        h = heuristic_cycles(llm_ir_path)
        if h is not None:
            est_pre = pct(pre,  h) + "(est)"
            est_llvm = pct(llvm, h) + "(est)"
        else:
            est_pre = "N/A"
            est_llvm = "N/A"

    llm_display = llm if isinstance(llm, int) else str(llm).split(":")[0]
    pre_vs_llvm = pct(pre, llvm)
    print(f"{pre},{llvm},{llm_display},{pre_vs_llvm},{real_pre_vs_llm},{real_llvm_vs_llm},{est_pre},{est_llvm}")


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2], sys.argv[3])


warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

csv_file = "/content/drive/MyDrive/mca_evaluation_log.csv"
print(f"\n{'='*100}")
print(f"STARTING COMPREHENSIVE EVALUATION ON {len(test_ds)} SAMPLES")
print(f"RESULTS LOGGED TO: {csv_file}")
print(f"{'='*100}\n")

with open(csv_file, 'w') as f:
    f.write("Row_ID,PRE_Cycles,LLVM_O3_Cycles,LLM_O3_Cycles,Pre_vs_LLVM,Pre_vs_LLM,LLVM_vs_LLM,Est_LLM_vs_Pre,Est_LLM_vs_LLVM\n")

HDR = f"{'Row ID':<7} | {'PRE':<8} | {'LLVM-O3':<8} | {'LLM-O3':<12} | {'Pre vs LLVM':<13} | {'Pre vs LLM':<12} | {'LLVM vs LLM':<13} | {'Est LLM vs PRE':<16} | {'Est LLM vs LLVM':<16}"
print(HDR)
print("-" * len(HDR))

for i, example in enumerate(test_ds):
    pre_ir = example["preprocessed_ir"]
    llvm_ir = example["o3_ir"]

    prompt = alpaca_prompt.format(
        "Given unoptimized LLVM IR, produce the equivalent O3-optimized IR.",
        pre_ir,
        "",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    if inputs['input_ids'].shape[1] > 24000:
        print(f'Skipping huge {inputs["input_ids"].shape[1]} token file...')
        continue
    outputs = model.generate(**inputs, max_new_tokens=8192,
                             use_cache=True, temperature=0.3, do_sample=True, top_p=0.9)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Response:\n" in response:
        llm_ir = response.split("### Response:\n")[-1]
    else:
        llm_ir = response

    with open("pre.ll", "w") as f:
        f.write(pre_ir)
    with open("llvm.ll", "w") as f:
        f.write(llvm_ir)
    with open("llm.ll", "w") as f:
        f.write(llm_ir)

    result = subprocess.run(["python", "mca_sanity_check.py", "pre.ll",
                            "llvm.ll", "llm.ll"], capture_output=True, text=True)
    csv_data = result.stdout.strip()

    row_string = f"{i},{csv_data}"
    with open(csv_file, 'a') as f:
        f.write(row_string + "\n")

    # cols: [0]row_id [1]pre [2]llvm [3]llm [4]pre_vs_llvm [5]pre_vs_llm [6]llvm_vs_llm [7]est_pre [8]est_llvm
    cols = row_string.split(",", 8)
    if len(cols) == 9:
        # For display, truncate Syntax_Error to just "Syntax_Error"
        def disp(v): return v.split(":")[0].strip(
        ) if v.startswith("Syntax_Error") else v
        print(f"{disp(cols[0]):<7} | {disp(cols[1]):<8} | {disp(cols[2]):<8} | {disp(cols[3]):<12} | {disp(cols[4]):<13} | {disp(cols[5]):<12} | {disp(cols[6]):<13} | {disp(cols[7]):<16} | {disp(cols[8]):<16}")
    else:
        print(f"{i:<7} | PARSE ERROR: {csv_data[:60]}")

print(f"\n{'='*100}")
print("EVALUATION COMPLETE - Logged to Google Drive.")
print(f"{'='*100}\n")


'''

====================================================================================================
STARTING COMPREHENSIVE EVALUATION ON 200 SAMPLES
RESULTS LOGGED TO: /content/drive/MyDrive/mca_evaluation_log.csv
====================================================================================================

Row ID  | PRE      | LLVM-O3  | LLM-O3       | Pre vs LLVM   | Pre vs LLM   | LLVM vs LLM   | Est LLM vs PRE   | Est LLM vs LLVM 
---------------------------------------------------------------------------------------------------------------------------------
0       | 16638    | Syntax_Error | Syntax_Error | N/A           | N/A          | N/A           | 95.06%(est)      | N/A(est)        
1       | 10000    | 9969     | Syntax_Error | 0.31%         | N/A          | N/A           | 96.74%(est)      | 96.73%(est)     
Skipping huge 97227 token file...
3       | 17173    | 11991    | Syntax_Error | 30.18%        | N/A          | N/A           | 96.59%(est)      | 95.12%(est)     
4       | 29593    | 27917    | Syntax_Error | 5.66%         | N/A          | N/A           | 96.95%(est)      | 96.76%(est)     
5       | 10263    | 10956    | Syntax_Error | -6.75%        | N/A          | N/A           | 95.34%(est)      | 95.64%(est)     
Skipping huge 52722 token file...
7       | 10282    | 11047    | Syntax_Error | -7.44%        | N/A          | N/A           | 95.15%(est)      | 95.48%(est)     
Skipping huge 27489 token file...
9       | 620      | 6907     | Syntax_Error | -1014.03%     | N/A          | N/A           | 82.90%(est)      | 98.47%(est)     
10      | 1162     | 1611     | Syntax_Error | -38.64%       | N/A          | N/A           | 2.93%(est)       | 29.98%(est)     
11      | 1215     | 4514     | Syntax_Error | -271.52%      | N/A          | N/A           | 91.85%(est)      | 97.81%(est)     
12      | 9158     | 9170     | Syntax_Error | -0.13%        | N/A          | N/A           | 98.20%(est)      | 98.20%(est)     
13      | 2337     | 2305     | Syntax_Error | 1.37%         | N/A          | N/A           | 91.57%(est)      | 91.45%(est)     
14      | 3778     | 3855     | Syntax_Error | -2.04%        | N/A          | N/A           | 96.53%(est)      | 96.60%(est)     
15      | 3754     | 3520     | Syntax_Error | 6.23%         | N/A          | N/A           | 96.03%(est)      | 95.77%(est)     
16      | 1694     | 1842     | Syntax_Error | -8.74%        | N/A          | N/A           | 97.76%(est)      | 97.94%(est)     
...
174     | 12421    | 12306    | Syntax_Error | 0.93%         | N/A          | N/A           | 92.67%(est)      | 92.61%(est)     
175     | 6941     | Syntax_Error | Syntax_Error | N/A           | N/A          | N/A           | 97.91%(est)      | N/A(est)        
176     | 4668     | 3007     | 3056         | 35.58%        | 34.53%       | -1.63%        | 34.53%           | -1.63%          
177     | 2318     | 9076     | Syntax_Error | -291.54%      | N/A          | N/A           | 70.92%(est)      | 92.57%(est) 
'''

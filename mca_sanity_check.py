import subprocess
import re
import sys
import os


# ── Skylake cycle costs per LLVM IR opcode ──────────────────────────
CYCLE_TABLE = {
    'add': 1, 'sub': 1, 'and': 1, 'or': 1, 'xor': 1, 'shl': 1, 'lshr': 1, 'ashr': 1,
    'icmp': 1, 'select': 1, 'zext': 1, 'sext': 1, 'trunc': 1, 'bitcast': 0,
    'phi': 0, 'alloca': 1, 'extractvalue': 1, 'insertvalue': 1, 'getelementptr': 1,
    'mul': 3,
    'udiv': 25, 'sdiv': 25, 'urem': 25, 'srem': 25,
    'fadd': 4, 'fsub': 4, 'fmul': 4, 'frem': 20, 'fdiv': 14,
    'fptoui': 3, 'fptosi': 3, 'uitofp': 3, 'sitofp': 3, 'fpext': 1, 'fptrunc': 1,
    'load': 4, 'store': 3, 'atomicrmw': 10, 'cmpxchg': 15,
    'br': 1, 'ret': 1, 'switch': 2, 'indirectbr': 3, 'unreachable': 0,
    'call': 10, 'invoke': 10, 'tail': 10,
}


def heuristic_cycles(filepath):
    """Estimate cycles from IR text when compilation fails."""
    try:
        with open(filepath, 'r', errors='replace') as f:
            lines = f.readlines()
    except Exception:
        return None

    opcode_re = re.compile(r'^\s+(?:%[\w.]+\s*=\s*)?(tail\s+)?([\w.]+)')
    total = 0
    phi_count = 0
    bb_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith((';', '!', 'source_filename',
                                                'target', 'attributes',
                                                'define', 'declare', '}')):
            continue
        if re.match(r'^[\w.][\w.]*:', stripped):
            bb_count += 1
            continue
        m = opcode_re.match(line)
        if not m:
            continue
        opcode = m.group(2).lower()
        total += CYCLE_TABLE.get(opcode, 0)
        if opcode == 'phi':
            phi_count += 1

    if bb_count > 0 and phi_count / bb_count > 0.3:
        total = int(total * 0.6 + total * 0.4 * 8)

    return total if total > 0 else None


def get_mca_cycles(filepath, skip_verify=False):
    """
    Try to get cpu cycles for the given IR file.

    Strategy:
    1. If skip_verify=False, run opt --passes=verify first (strict mode).
       If that fails → return Syntax_Error.
    2. If skip_verify=True, skip verify and go straight to llc.
       If llc fails → fall back to heuristic estimator.
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
        result = subprocess.run(
            ["llc", "-O0", filepath, "-o", asm],
            capture_output=True
        )
        if result.returncode != 0:
            err = result.stderr.decode()[:120].strip().replace(',', ';')
            return f"Syntax_Error: LLC_Failed: {err}"

        with open(asm, 'r') as f_in, open(clean_asm, 'w') as f_out:
            for line in f_in:
                if not line.strip().startswith('.'):
                    f_out.write(line)

        mca = subprocess.run(
            ["llvm-mca", "-mcpu=skylake", clean_asm],
            capture_output=True, text=True, check=True
        )
        m = re.search(r"Total Cycles:\s+(\d+)", mca.stdout)
        if os.path.exists(asm):
            os.remove(asm)
        if os.path.exists(clean_asm):
            os.remove(clean_asm)
        return int(m.group(1)) if m else "MCA_Failed"

    except Exception as e:
        return "MCA_Error"


def pct(base, other):
    if not isinstance(base, int) or not isinstance(other, int):
        return "N/A"
    return f"{((base - other) / max(base, 1)) * 100:.2f}%"


def evaluate(pre_ir_path, llvm_ir_path, llm_ir_path):
    """Full 3-file evaluation used from the Colab notebook."""
    pre = get_mca_cycles(pre_ir_path,  skip_verify=False)
    # skip verify for O3 IR
    llvm = get_mca_cycles(llvm_ir_path, skip_verify=True)
    llm = get_mca_cycles(llm_ir_path,  skip_verify=False)

    pre_vs_llvm = pct(pre,  llvm)
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
    print(f"{pre},{llvm},{llm_display},{pre_vs_llvm},{real_pre_vs_llm},{real_llvm_vs_llm},{est_pre},{est_llvm}")


def compare_two(pre_ir_path, opt_ir_path):
    """Simple 2-file comparison — for local testing."""
    print("=" * 60)
    print("LLVM Code Analysis")
    print("=" * 60)

    print(f"\nPRE IR  ({pre_ir_path}):")
    pre_cycles = get_mca_cycles(pre_ir_path, skip_verify=False)
    print(f"  Result: {pre_cycles}")

    print(f"\nO3 IR   ({opt_ir_path}) [skipping opt --passes=verify]:")
    opt_cycles = get_mca_cycles(opt_ir_path, skip_verify=True)
    print(f"  Result: {opt_cycles}")

    print("\n" + "=" * 60)
    if isinstance(pre_cycles, int) and isinstance(opt_cycles, int):
        diff = pre_cycles - opt_cycles
        pct_diff = (diff / pre_cycles) * 100
        if diff > 0:
            print(
                f"SUCCESS: O3 is faster by {diff} cycles ({pct_diff:.2f}% reduction)")
        elif diff == 0:
            print("TIE: Both versions take the same number of cycles.")
        else:
            print(
                f"SLOWER: O3 is slower by {abs(diff)} cycles ({abs(pct_diff):.2f}% increase)")
    else:
        print(f"PRE: {pre_cycles}")
        print(f"O3:  {opt_cycles}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare_two(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        evaluate(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage:")
        print("  python mca_sanity_check.py <pre.ll> <opt.ll>              (2-file test)")
        print(
            "  python mca_sanity_check.py <pre.ll> <llvm.ll> <llm.ll>   (3-file evaluation)")
        sys.exit(1)

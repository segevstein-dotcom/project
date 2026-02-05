#!/usr/bin/env python3
"""
Root cause analysis: Why does C output differ from golden?
"""

import os
import numpy as np

DATA_FOLDER = "data/quantized"

# Load quantization params
with open(os.path.join(DATA_FOLDER, "global_params.txt")) as f:
    lines = [l.strip() for l in f.readlines()]
    hidden_dim = int(lines[1])
    num_vecs = int(lines[2])
    global_s = float(lines[3])
    global_zp = int(lines[4])

# Load alpha factors
with open(os.path.join(DATA_FOLDER, "alpha_factors.txt")) as f:
    lines = [l.strip() for l in f.readlines()]
    alpha_factors = []
    for line in lines[1:]:
        if line and not line.startswith('#'):
            alpha_factors.append(int(line))

# Load quantized vector 0
with open(os.path.join(DATA_FOLDER, "vector_000.txt")) as f:
    lines = [l.strip() for l in f.readlines()]
    vec0_quantized = []
    for line in lines[1:]:
        if line and not line.startswith('#'):
            vec0_quantized.append(int(line))

# Load C output
with open(os.path.join(DATA_FOLDER, "c_output_vec000.txt")) as f:
    lines = [l.strip() for l in f.readlines()]
    c_output = []
    capture = False
    for line in lines:
        if "Output:" in line:
            capture = True
            continue
        if capture and line and not line.startswith("#"):
            try:
                c_output.append(float(line))
            except:
                pass

# Load golden output
with open(os.path.join(DATA_FOLDER, "golden_ref_vec000.txt")) as f:
    lines = [l.strip() for l in f.readlines()]
    golden_mean = 0
    golden_var = 0
    golden_output = []
    for line in lines:
        if "ORIGINAL Mean:" in line:
            golden_mean = float(line.split(": ")[1])
        if "ORIGINAL Variance:" in line:
            golden_var = float(line.split(": ")[1])
        if "Output:" in line:
            idx = lines.index(line)
            for l in lines[idx+1:]:
                if l and not l.startswith("#"):
                    try:
                        golden_output.append(float(l))
                    except:
                        pass
            break

print("="*80)
print("ROOT CAUSE ANALYSIS: C vs Golden Mismatch")
print("="*80)

print(f"\n1. QUANTIZATION SETUP:")
print(f"   Hidden dim: {hidden_dim}, Alpha factors: {len(set(alpha_factors))} unique values {set(alpha_factors)}")
print(f"   Global S: {global_s:.10f}, Global ZP: {global_zp}")
print(f"   Min Alpha: {min(alpha_factors)}, Max Alpha: {max(alpha_factors)}")

print(f"\n2. GOLDEN REFERENCE (Python):")
print(f"   Uses ORIGINAL input statistics (before quantization):")
print(f"      Mean: {golden_mean:.6f}")
print(f"      Variance: {golden_var:.6f}")
print(f"   Expected output stats: {len(golden_output)} values, mean={np.mean(golden_output):.6f}")

print(f"\n3. C IMPLEMENTATION:")
print(f"   Uses INTEGER-DOMAIN STATISTICS (from quantized input + compression):")
print(f"      Stage1: Computes Ex (sum) and Ex2 (sum of compressed squares)")
print(f"      Stage2: Normalizes using mean_hw = Ex/C and var_hw = (Ex2<<4)/C - mean_hw^2")
print(f"   Actual C output: {len(c_output)} values, mean={np.mean(c_output):.6f}")

print(f"\n4. ERROR ANALYSIS:")
errors = np.array([abs(c_output[i] - golden_output[i]) for i in range(len(c_output))])
print(f"   MAE: {np.mean(errors):.6f}")
print(f"   MSE: {np.mean(errors**2):.6f}")
print(f"   Max error: {np.max(errors):.6f} at channel {np.argmax(errors)}")
print(f"   Median error: {np.median(errors):.6f}")

# Analyze by alpha
print(f"\n5. ERROR vs ALPHA FACTOR:")
for alpha_val in sorted(set(alpha_factors)):
    indices = [i for i, a in enumerate(alpha_factors) if a == alpha_val and i < len(c_output)]
    if indices:
        alpha_errors = [errors[i] for i in indices]
        print(f"   Alpha {alpha_val:2d} ({len(indices):3d} channels): mean_err={np.mean(alpha_errors):.6f}, max_err={np.max(alpha_errors):.6f}")

# High error channels
print(f"\n6. TOP 15 HIGH-ERROR CHANNELS:")
sorted_indices = np.argsort(errors)[::-1][:15]
for rank, idx in enumerate(sorted_indices, 1):
    alpha = alpha_factors[idx]
    c_val = c_output[idx]
    g_val = golden_output[idx]
    print(f"   {rank:2d}. Ch {idx:3d} (alpha={alpha:2d}): C={c_val:8.6f}, Golden={g_val:8.6f}, Err={errors[idx]:.6f}")

print(f"\n7. ROOT CAUSE HYPOTHESIS:")
print(f"   The C code uses INTEGER-DOMAIN statistics in Stage1:")
print(f"      - It reconstructs xi = quantized[i] - ZP")
print(f"      - It compresses |xi| via dynamic_compress() and squares using LUT")
print(f"      - It shifts by (alpha - min_alpha) to create integer Ex and Ex2")
print(f"      - The LUT approximation introduces ERROR in the variance computation")
print(f"")
print(f"   But the golden reference uses ORIGINAL float statistics:")
print(f"      - It normalizes using the true original mean/variance (BEFORE quantization)")
print(f"      - This is the mathematically correct approach for reference")
print(f"")
print(f"   => The MISMATCH is FUNDAMENTAL:")
print(f"      - If C uses quantized stats, it will always differ from golden")
print(f"      - The golden should be generated using the SAME stats as C")
print(f"      - OR C should use original float inputs (not feasible in HW)")
print(f"")
print(f"   => RESOLUTION:")
print(f"      - Change Python golden to use the QUANTIZED/RECONSTRUCTED input stats")
print(f"      - OR simulate the LUT compression in Python to match C behavior")

print(f"\n" + "="*80)

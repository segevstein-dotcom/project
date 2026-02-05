#!/usr/bin/env python3
"""
Diagnostic script to identify root causes of C vs golden mismatch.
Checks: compression errors, LUT approximations, normalization, and affine transform.
"""

import os
import numpy as np
import math

# Load all generated files
DATA_FOLDER = "data/quantized"

def load_file(fname):
    with open(os.path.join(DATA_FOLDER, fname)) as f:
        lines = [l.strip() for l in f.readlines()]
    return lines

def parse_int_list(lines):
    data = []
    for line in lines:
        if line.startswith('#') or line == '':
            continue
        try:
            data.append(int(line))
        except:
            pass
    return data

def parse_float_list(lines):
    data = []
    for line in lines:
        if line.startswith('#') or line == '':
            continue
        try:
            data.append(float(line))
        except:
            pass
    return data

# Load quantization parameters
lines_global = load_file("global_params.txt")
hidden_dim = int(lines_global[1])
num_vecs = int(lines_global[2])
global_s = float(lines_global[3])
global_zp = int(lines_global[4])

# Load alpha factors
lines_alpha = load_file("alpha_factors.txt")
alpha_factors = [int(l) for l in lines_alpha[1:] if not l.startswith('#')]

# Load vector 0 (quantized)
vec0_lines = load_file("vector_000.txt")
vec0_quantized = parse_int_list(vec0_lines)

# Load golden reference (python generated)
golden_lines = load_file("golden_ref_vec000.txt")
golden_output = []
capture = False
for line in golden_lines:
    if line.startswith("Output:"):
        capture = True
        continue
    if capture and line and not line.startswith("#"):
        try:
            golden_output.append(float(line))
        except:
            pass

# Load C output
c_lines = load_file("c_output_vec000.txt")
c_output = []
capture = False
for line in c_lines:
    if line.startswith("Output:"):
        capture = True
        continue
    if capture and line and not line.startswith("#"):
        try:
            c_output.append(float(line))
        except:
            pass

print("="*80)
print("DIAGNOSTIC: Root Cause Analysis of C vs Golden Mismatch")
print("="*80)

print(f"\n1. QUANTIZATION PARAMETERS:")
print(f"   Hidden Dim: {hidden_dim}")
print(f"   Global S: {global_s:.10f}")
print(f"   Global ZP: {global_zp}")
print(f"   Unique Alpha values: {set(alpha_factors)}")

# ============================================================================
# STAGE 1: Reconstruct inputs and check compression/LUT approximations
# ============================================================================
print(f"\n2. STAGE 1: INPUT RECONSTRUCTION & COMPRESSION ANALYSIS")

# Load gamma/beta
gamma_q_lines = load_file("gamma_quantized.txt")
beta_q_lines = load_file("beta_quantized.txt")

# Load gamma/beta quantization parameters from file headers
gamma_q_lines = load_file("gamma_quantized.txt")
beta_q_lines = load_file("beta_quantized.txt")

# Parse header comments for scale/zp
def extract_quant_params(lines):
    for line in lines:
        if "Scale:" in line:
            scale = float(line.split("Scale: ")[1].split(",")[0])
            zp = int(line.split("ZP: ")[1])
            return scale, zp
    return 1.0, 0

gamma_q_scale, gamma_q_zp = extract_quant_params(gamma_q_lines)
beta_q_scale, beta_q_zp = extract_quant_params(beta_q_lines)

# Extract values (skip headers and first count line)
def extract_quant_vals(lines):
    count_found = False
    skip_next = 2  # skip scale and zp lines
    vals = []
    for line in lines:
        if line.isdigit() or (line.replace('-','').isdigit()):
            if not count_found:
                count_found = True
                continue
            if skip_next > 0:
                skip_next -= 1
                continue
            try:
                vals.append(int(line))
            except:
                pass
    return vals[:384]  # Take first 384

gamma_q_vals = extract_quant_vals(gamma_q_lines)
beta_q_vals = extract_quant_vals(beta_q_lines)

# Reconstruct dequantized values (float)
reconstructed_input = np.array([
    (int(vec0_quantized[i]) - global_zp) * (global_s * (2 ** alpha_factors[i]))
    for i in range(hidden_dim)
])

print(f"   Reconstructed input range: [{np.min(reconstructed_input):.6f}, {np.max(reconstructed_input):.6f}]")
print(f"   Reconstructed input mean: {np.mean(reconstructed_input):.6f}")
print(f"   Reconstructed input variance: {np.var(reconstructed_input):.6f}")

# Now check dynamic compression (HW approximation)
SQUARE_LUT = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]

def dynamic_compress(x):
    """Simulate C's dynamic_compress"""
    abs_x = abs(x)
    if abs_x >= 64:
        compressed = int(abs_x >> 4)
        shift = 1
    else:
        compressed = int(abs_x >> 2)
        shift = 0
    if compressed > 15:
        compressed = 15
    sq = SQUARE_LUT[compressed] << (4 * shift)
    return abs_x * abs_x, sq  # true_sq, approx_sq

# Analyze compression errors for all channels
min_alpha = min(alpha_factors)
compression_errors = []
for i in range(hidden_dim):
    xi = int(vec0_quantized[i]) - global_zp
    true_sq = xi * xi
    approx_sq, _ = dynamic_compress(abs(xi))
    err = abs(true_sq - approx_sq)
    if err > 0:
        compression_errors.append((i, xi, true_sq, approx_sq, err))

if compression_errors:
    print(f"\n   Compression Errors (Top 10):")
    sorted_errors = sorted(compression_errors, key=lambda x: x[4], reverse=True)[:10]
    for idx, xi, true_sq, approx_sq, err in sorted_errors:
        pct = (err / true_sq * 100) if true_sq > 0 else 0
        print(f"      Ch {idx:3d}: xi={xi:4d}, true_sq={true_sq:6d}, approx_sq={approx_sq:6d}, error={err:6d} ({pct:.2f}%)")
else:
    print(f"   No compression errors detected!")

# ============================================================================
# STAGE 2: Compare C normalization vs expected
# ============================================================================
print(f"\n3. STAGE 2: NORMALIZATION & AFFINE TRANSFORM")

# The C code uses:
# mean_hw = Ex / C
# var_hw = (Ex2 << 4) / C - mean_hw^2
# Then applies: xi_norm = (xi_scaled - mean_hw) * stdinv_q >> STDINV_Q

# Golden uses original float mean/variance
# Let's check the mismatch in normalization

golden_mean_val = float([l.split(": ")[1] for l in golden_lines if l.startswith("ORIGINAL Mean:")][0])
golden_var_val = float([l.split(": ")[1] for l in golden_lines if l.startswith("ORIGINAL Variance:")][0])

print(f"   Golden (Original) Mean: {golden_mean_val:.6f}")
print(f"   Golden (Original) Variance: {golden_var_val:.6f}")

# Approximate what C computes from integer operations
# For Stage1: Ex is accumulated with shifts based on min_alpha
# Here we'd need to simulate the exact C stage1 computation
print(f"   Min Alpha: {min_alpha}")

# ============================================================================
# ANALYZE PER-CHANNEL ERRORS
# ============================================================================
print(f"\n4. PER-CHANNEL ERROR ANALYSIS")

if len(c_output) == len(golden_output):
    errors = np.array([abs(c_output[i] - golden_output[i]) for i in range(len(c_output))])
    
    print(f"   Total channels: {len(errors)}")
    print(f"   MAE: {np.mean(errors):.6f}")
    print(f"   MSE: {np.mean(errors**2):.6f}")
    print(f"   Max error: {np.max(errors):.6f}")
    print(f"   Median error: {np.median(errors):.6f}")
    
    # Identify pattern
    high_error_mask = errors > 0.5
    print(f"\n   Channels with error > 0.5: {np.sum(high_error_mask)}")
    high_error_indices = np.where(high_error_mask)[0]
    if len(high_error_indices) > 0:
        print(f"      Top 10 high-error channels:")
        sorted_idx = np.argsort(errors)[::-1][:10]
        for idx in sorted_idx:
            alpha = alpha_factors[idx]
            print(f"         Ch {idx:3d}: alpha={alpha:2d}, c={c_output[idx]:8.6f}, golden={golden_output[idx]:8.6f}, err={errors[idx]:.6f}")
    
    # Check if high errors correlate with certain alpha values
    print(f"\n   Error by Alpha Value:")
    for alpha_val in sorted(set(alpha_factors)):
        mask = np.array([alpha_factors[i] == alpha_val for i in range(len(alpha_factors))])
        if np.sum(mask) > 0:
            alpha_errors = errors[mask]
            print(f"      Alpha {alpha_val:2d}: {np.sum(mask):3d} channels, mean_err={np.mean(alpha_errors):.6f}, max_err={np.max(alpha_errors):.6f}")

# ============================================================================
# HYPOTHESIS: Check if Stage1 mean/variance mismatch is the issue
# ============================================================================
print(f"\n5. HYPOTHESIS: Is Stage1 Stats Mismatch the Root Cause?")

# The issue may be that:
# - Golden uses ORIGINAL float input stats
# - C computes stats from QUANTIZED reconstructed input (with compression/LUT loss)

# So the normalization in Stage2 is based on different statistics!
# This would cause every channel to be shifted/scaled differently.

print(f"   Golden uses ORIGINAL input mean/variance for normalization")
print(f"   C uses INTEGER-DOMAIN Ex/Ex2 (with compression loss) for normalization")
print(f"   => All C outputs are biased relative to golden due to different means!")

print(f"\n" + "="*80)

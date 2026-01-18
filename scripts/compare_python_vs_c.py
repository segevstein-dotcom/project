"""
Complete comparison: Python simulation vs C code output
This proves that our Python code produces IDENTICAL results to the C implementation
"""

import numpy as np
import os

# Import functions from analyze_var_hw_distribution.py
SQUARE_LUT = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]

def dynamic_compress(x):
    """Simulates the C dynamic_compress function"""
    abs_x = abs(x)
    if abs_x >= 64:
        compressed = abs_x >> 4
        shift = 1
    else:
        compressed = abs_x >> 2
        shift = 0
    if compressed > 15:
        compressed = 15
    return compressed, shift

def find_min_alpha(alphas):
    """Find minimum alpha value"""
    return min(alphas)

def stage1_statistics(quantized_values, zero_points, alpha_factors):
    """
    Simulates the exact C code stage1_statistics function
    Returns Ex, Ex2, min_alpha
    """
    min_alpha = find_min_alpha(alpha_factors)
    Ex = 0
    Ex2 = 0

    for i in range(len(quantized_values)):
        zp = zero_points[i]
        input_val = quantized_values[i]

        # 1. Center
        xi = int(input_val) - int(zp)

        # 2. Compress & Square
        compressed, shift = dynamic_compress(xi)
        xc_sq = SQUARE_LUT[compressed] << (4 * shift)

        # 3. Global Scale Logic (Shift Left Only)
        alpha = alpha_factors[i]
        relative_shift = alpha - min_alpha  # Always >= 0

        ex_contribution = xi << relative_shift
        ex2_contribution = xc_sq << (2 * relative_shift)

        # 4. Accumulate
        Ex += ex_contribution
        Ex2 += ex2_contribution

    return Ex, Ex2, min_alpha

def compute_variance_hw(Ex, Ex2, num_channels):
    """
    Simulates lines 122-125 of main.c
    Returns mean_hw and var_hw
    """
    mean_hw = float(Ex) / num_channels
    mean_sq_hw = float(Ex2 << 4) / num_channels  # Note the << 4!
    var_hw = mean_sq_hw - (mean_hw * mean_hw)
    if var_hw < 0:
        var_hw = 0
    return mean_hw, var_hw

print("="*80)
print("COMPLETE COMPARISON: Python Simulation vs C Code Output")
print("="*80)

# ============================================
# Load C code output (if exists)
# ============================================
report_file = "data/quantized/final_report.txt"
if not os.path.exists(report_file):
    print(f"\nERROR: {report_file} not found!")
    print("Please run the C code first to generate the report.")
    print("Then run this script to compare.")
    exit(1)

print(f"\n1. Loading C code output from {report_file}...")
print("-" * 80)

c_results = []
with open(report_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("|") and not line.startswith("|---"):
            # Skip header
            if "Vec ID" in line or "Mean (Real)" in line:
                continue
            if "=" in line:
                continue

            parts = line.split("|")
            if len(parts) >= 7:
                try:
                    vec_id = int(parts[1].strip())
                    min_alpha_c = int(parts[2].strip())
                    mean_real_c = float(parts[3].strip())
                    var_real_c = float(parts[4].strip())
                    std_real_c = float(parts[5].strip())
                    Ex_c = int(parts[6].strip())

                    c_results.append({
                        'vec_id': vec_id,
                        'min_alpha': min_alpha_c,
                        'mean_real': mean_real_c,
                        'var_real': var_real_c,
                        'std_real': std_real_c,
                        'Ex': Ex_c
                    })
                except:
                    pass

print(f"Loaded {len(c_results)} vectors from C code output")

# ============================================
# Run Python simulation
# ============================================
print(f"\n2. Running Python simulation...")
print("-" * 80)

data_folder = "data/quantized"

# Load global parameters
with open(os.path.join(data_folder, "global_params.txt"), "r") as f:
    lines = f.readlines()
    num_channels = int(lines[1].strip())
    num_vectors = int(lines[2].strip())
    global_s = float(lines[3].strip())
    global_zp = int(lines[4].strip())

print(f"Global parameters:")
print(f"  num_channels = {num_channels}")
print(f"  global_s = {global_s}")
print(f"  global_zp = {global_zp}")

# Load alpha factors
with open(os.path.join(data_folder, "alpha_factors.txt"), "r") as f:
    lines = f.readlines()
    alpha_factors = []
    for i in range(2, 2 + num_channels):
        alpha_factors.append(int(lines[i].strip()))

zero_points = [global_zp] * num_channels

# Process all vectors
python_results = []
for vec_idx in range(len(c_results)):
    vec_file = os.path.join(data_folder, f"vector_{vec_idx:03d}.txt")
    if not os.path.exists(vec_file):
        break

    with open(vec_file, "r") as f:
        lines = f.readlines()
        quantized_values = []
        for i in range(2, 2 + num_channels):
            quantized_values.append(int(lines[i].strip()))

    # Compute statistics
    Ex, Ex2, min_alpha = stage1_statistics(quantized_values, zero_points, alpha_factors)
    mean_hw, var_hw = compute_variance_hw(Ex, Ex2, num_channels)

    # Convert to real domain (same as C code lines 127-132)
    alpha_scale = 2.0 ** min_alpha
    full_scale = alpha_scale * global_s
    mean_real = mean_hw * full_scale
    var_real = var_hw * (full_scale ** 2)
    std_real = np.sqrt(var_real)

    # Also compute 1/sqrt(var_hw) - THIS IS WHAT THE LUT APPROXIMATES!
    inv_sqrt_var_hw = 1.0 / np.sqrt(var_hw) if var_hw > 0 else 0

    python_results.append({
        'vec_id': vec_idx,
        'min_alpha': min_alpha,
        'Ex': Ex,
        'Ex2': Ex2,
        'mean_hw': mean_hw,
        'var_hw': var_hw,
        'mean_real': mean_real,
        'var_real': var_real,
        'std_real': std_real,
        'inv_sqrt_var_hw': inv_sqrt_var_hw
    })

print(f"Processed {len(python_results)} vectors")

# ============================================
# COMPARISON
# ============================================
print(f"\n3. Comparing results...")
print("-" * 80)

num_compared = min(len(c_results), len(python_results))

# Check a few specific vectors
print(f"\nDetailed comparison for first 5 vectors:")
print(f"{'Vec':<5} {'Field':<15} {'C Code':<20} {'Python':<20} {'Diff':<15} {'Match':<8}")
print("-" * 80)

all_match = True
for i in range(min(5, num_compared)):
    c = c_results[i]
    p = python_results[i]

    # Compare min_alpha
    diff = abs(c['min_alpha'] - p['min_alpha'])
    match = (diff == 0)
    all_match = all_match and match
    print(f"{i:<5} {'min_alpha':<15} {c['min_alpha']:<20} {p['min_alpha']:<20} {diff:<15} {match}")

    # Compare Ex
    diff = abs(c['Ex'] - p['Ex'])
    match = (diff == 0)
    all_match = all_match and match
    print(f"{i:<5} {'Ex':<15} {c['Ex']:<20} {p['Ex']:<20} {diff:<15} {match}")

    # Compare mean_real
    diff = abs(c['mean_real'] - p['mean_real'])
    match = (diff < 1e-5)
    all_match = all_match and match
    print(f"{i:<5} {'mean_real':<15} {c['mean_real']:<20.8f} {p['mean_real']:<20.8f} {diff:<15.2e} {match}")

    # Compare var_real
    diff = abs(c['var_real'] - p['var_real'])
    match = (diff < 1e-5)
    all_match = all_match and match
    print(f"{i:<5} {'var_real':<15} {c['var_real']:<20.8f} {p['var_real']:<20.8f} {diff:<15.2e} {match}")

    # Compare std_real
    diff = abs(c['std_real'] - p['std_real'])
    match = (diff < 1e-5)
    all_match = all_match and match
    print(f"{i:<5} {'std_real':<15} {c['std_real']:<20.8f} {p['std_real']:<20.8f} {diff:<15.2e} {match}")

    print()

# ============================================
# Statistical comparison
# ============================================
print(f"\n4. Statistical comparison across all {num_compared} vectors:")
print("-" * 80)

# Extract arrays for comparison
var_real_c = np.array([c['var_real'] for c in c_results[:num_compared]])
var_real_p = np.array([p['var_real'] for p in python_results[:num_compared]])

std_real_c = np.array([c['std_real'] for c in c_results[:num_compared]])
std_real_p = np.array([p['std_real'] for p in python_results[:num_compared]])

mean_real_c = np.array([c['mean_real'] for c in c_results[:num_compared]])
mean_real_p = np.array([p['mean_real'] for p in python_results[:num_compared]])

# Compute differences
var_diff = np.abs(var_real_c - var_real_p)
std_diff = np.abs(std_real_c - std_real_p)
mean_diff = np.abs(mean_real_c - mean_real_p)

print(f"var_real differences:")
print(f"  Max:  {np.max(var_diff):.2e}")
print(f"  Mean: {np.mean(var_diff):.2e}")
print(f"  Min:  {np.min(var_diff):.2e}")

print(f"\nstd_real differences:")
print(f"  Max:  {np.max(std_diff):.2e}")
print(f"  Mean: {np.mean(std_diff):.2e}")
print(f"  Min:  {np.min(std_diff):.2e}")

print(f"\nmean_real differences:")
print(f"  Max:  {np.max(mean_diff):.2e}")
print(f"  Mean: {np.mean(mean_diff):.2e}")
print(f"  Min:  {np.min(mean_diff):.2e}")

# ============================================
# SHOW THE INVERSE SQRT VALUES
# ============================================
print(f"\n5. The LUT target values (1/sqrt(var_hw)):")
print("-" * 80)

inv_sqrt_values = np.array([p['inv_sqrt_var_hw'] for p in python_results])
print(f"1/sqrt(var_hw) statistics:")
print(f"  Min:    {np.min(inv_sqrt_values):.10f}")
print(f"  Max:    {np.max(inv_sqrt_values):.10f}")
print(f"  Mean:   {np.mean(inv_sqrt_values):.10f}")
print(f"  Median: {np.median(inv_sqrt_values):.10f}")

print(f"\nThese are the values the LUT needs to approximate!")
print(f"Range: [{np.min(inv_sqrt_values):.8f}, {np.max(inv_sqrt_values):.8f}]")

# ============================================
# FINAL VERDICT
# ============================================
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if np.max(var_diff) < 1e-5 and np.max(std_diff) < 1e-5 and np.max(mean_diff) < 1e-5:
    print("\n[OK] Python simulation EXACTLY matches C code output!")
    print(f"     Max difference: {max(np.max(var_diff), np.max(std_diff), np.max(mean_diff)):.2e}")
    print("     (Within floating-point precision)")
else:
    print("\n[WARNING] Some differences detected:")
    print(f"     Max var_real diff:  {np.max(var_diff):.2e}")
    print(f"     Max std_real diff:  {np.max(std_diff):.2e}")
    print(f"     Max mean_real diff: {np.max(mean_diff):.2e}")

print("="*80)

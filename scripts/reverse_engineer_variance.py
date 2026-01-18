"""
REVERSE ENGINEERING: Going backwards from var_hw to var_real

This script demonstrates the EXACT mathematical relationship between:
- var_hw (hardware domain variance - what we compute in C)
- var_real (real domain variance - the actual variance of original float data)

Based on lines 127-132 in main.c
"""

import numpy as np
import os

print("="*80)
print("REVERSE ENGINEERING: var_hw -> var_real")
print("="*80)

# ============================================
# Load var_hw values from analysis
# ============================================
print("\n1. Loading var_hw values from previous analysis...")
print("-" * 80)

var_hw_list = []
with open("data/lut/var_hw_analysis.txt", 'r') as f:
    lines = f.readlines()
    reading_data = False
    for line in lines:
        if line.startswith("All var_hw values:"):
            reading_data = True
            continue
        if reading_data and ',' in line:
            parts = line.strip().split(',')
            if len(parts) == 2:
                var_hw_list.append(float(parts[1]))

var_hw_array = np.array(var_hw_list)
print(f"Loaded {len(var_hw_array)} var_hw values")
print(f"var_hw range: [{np.min(var_hw_array):.2f}, {np.max(var_hw_array):.2f}]")

# ============================================
# Load global_s and min_alpha
# ============================================
print("\n2. Loading global_s from quantized data...")
print("-" * 80)

data_folder = "data/quantized"
with open(os.path.join(data_folder, "global_params.txt"), "r") as f:
    lines = f.readlines()
    num_channels = int(lines[1].strip())
    num_vectors = int(lines[2].strip())
    global_s = float(lines[3].strip())
    global_zp = int(lines[4].strip())

print(f"global_s = {global_s}")
print(f"num_channels = {num_channels}")
print(f"num_vectors = {num_vectors}")

# Load alpha factors to get min_alpha
with open(os.path.join(data_folder, "alpha_factors.txt"), "r") as f:
    lines = f.readlines()
    alpha_factors = []
    for i in range(2, 2 + num_channels):
        alpha_factors.append(int(lines[i].strip()))

min_alpha = min(alpha_factors)
print(f"min_alpha = {min_alpha}")

# ============================================
# THE MATHEMATICAL FORMULA (from main.c lines 127-132)
# ============================================
print("\n3. Applying reverse transformation...")
print("-" * 80)
print("\nFrom main.c (lines 127-132):")
print("  float alpha_scale = powf(2.0f, (float)min_alpha);")
print("  float full_scale = alpha_scale * data.global_s;")
print("  float mean_real = mean_hw * full_scale;")
print("  float var_real = var_hw * (full_scale * full_scale);  // variance scales by s^2")
print()

# Calculate the scaling factors
alpha_scale = 2.0 ** min_alpha
full_scale = alpha_scale * global_s

print(f"Calculated scaling factors:")
print(f"  alpha_scale = 2^{min_alpha} = {alpha_scale}")
print(f"  full_scale = alpha_scale * global_s = {alpha_scale} * {global_s} = {full_scale}")
print()

# The key formula:
# var_real = var_hw * (full_scale)^2
full_scale_squared = full_scale ** 2

print(f"Variance scaling factor:")
print(f"  full_scale^2 = {full_scale_squared}")
print()

# ============================================
# Apply reverse transformation
# ============================================
var_real_array = var_hw_array * full_scale_squared

print(f"Results:")
print(f"  var_hw  range: [{np.min(var_hw_array):.2f}, {np.max(var_hw_array):.2f}]")
print(f"  var_real range: [{np.min(var_real_array):.6f}, {np.max(var_real_array):.6f}]")
print()

# ============================================
# Statistical analysis of var_real
# ============================================
print("\n4. Statistical analysis of recovered var_real:")
print("-" * 80)
print(f"Min:     {np.min(var_real_array):.8f}")
print(f"Max:     {np.max(var_real_array):.8f}")
print(f"Mean:    {np.mean(var_real_array):.8f}")
print(f"Median:  {np.median(var_real_array):.8f}")
print(f"Std:     {np.std(var_real_array):.8f}")
print()
print(f"Percentiles:")
print(f"  1%:  {np.percentile(var_real_array, 1):.8f}")
print(f"  5%:  {np.percentile(var_real_array, 5):.8f}")
print(f"  25%: {np.percentile(var_real_array, 25):.8f}")
print(f"  50%: {np.percentile(var_real_array, 50):.8f}")
print(f"  75%: {np.percentile(var_real_array, 75):.8f}")
print(f"  95%: {np.percentile(var_real_array, 95):.8f}")
print(f"  99%: {np.percentile(var_real_array, 99):.8f}")

# ============================================
# Standard deviation in real domain
# ============================================
std_real_array = np.sqrt(var_real_array)

print(f"\nStandard deviation in real domain:")
print(f"  std_real range: [{np.min(std_real_array):.8f}, {np.max(std_real_array):.8f}]")
print(f"  std_real mean:  {np.mean(std_real_array):.8f}")

# ============================================
# Verify against C code output (if available)
# ============================================
print("\n5. Verification against C code output (if available):")
print("-" * 80)

report_file = "data/final_report.txt"
if os.path.exists(report_file):
    print(f"Reading {report_file}...")

    var_real_from_c = []
    with open(report_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("|") and not line.startswith("|---") and "Variance (Real)" not in line:
                parts = line.split("|")
                if len(parts) >= 5:
                    try:
                        # Extract variance from column 4 (index 4)
                        var_str = parts[4].strip()
                        var_val = float(var_str)
                        var_real_from_c.append(var_val)
                    except:
                        pass

    if len(var_real_from_c) > 0:
        var_real_from_c = np.array(var_real_from_c)
        print(f"\nLoaded {len(var_real_from_c)} variance values from C code output")
        print(f"C code var_real range: [{np.min(var_real_from_c):.6f}, {np.max(var_real_from_c):.6f}]")

        # Compare
        if len(var_real_from_c) == len(var_real_array):
            diff = np.abs(var_real_array - var_real_from_c)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)

            print(f"\nComparison with C code:")
            print(f"  Max difference:  {max_diff:.10e}")
            print(f"  Mean difference: {mean_diff:.10e}")

            if max_diff < 1e-6:
                print("\n  [OK] Python reverse engineering matches C code perfectly!")
            else:
                print(f"\n  [WARNING] Some differences detected (max={max_diff})")
        else:
            print(f"\n  [INFO] Number of values mismatch: C has {len(var_real_from_c)}, Python has {len(var_real_array)}")
    else:
        print("  [INFO] Could not parse variance values from report")
else:
    print(f"  [INFO] {report_file} not found - skipping verification")
    print("  Run the C code to generate this file for verification")

# ============================================
# THE KEY INSIGHT
# ============================================
print("\n" + "="*80)
print("KEY INSIGHT: The Scaling Relationship")
print("="*80)
print()
print(f"var_hw is {full_scale_squared:.2f}x LARGER than var_real!")
print()
print("This is because of TWO scaling factors:")
print(f"  1. min_alpha scaling:  2^(2*{min_alpha}) = {2**(2*min_alpha):.0f}x")
print(f"  2. global_s scaling:   (1/{global_s})^2 = {(1/global_s)**2:.2f}x")
print(f"  Total:                 {full_scale_squared:.2f}x")
print()
print("Formula:")
print(f"  var_real = var_hw / {full_scale_squared:.2f}")
print(f"  var_real = var_hw * {1/full_scale_squared:.10e}")
print()
print("Example:")
sample_var_hw = var_hw_array[0]
sample_var_real = var_real_array[0]
print(f"  var_hw  = {sample_var_hw:.2f}")
print(f"  var_real = {sample_var_hw:.2f} / {full_scale_squared:.2f} = {sample_var_real:.8f}")
print("="*80)

# ============================================
# Save results
# ============================================
output_file = "data/lut/reverse_engineering_results.txt"
with open(output_file, 'w') as f:
    f.write("REVERSE ENGINEERING: var_hw -> var_real\n")
    f.write("="*80 + "\n\n")

    f.write("Scaling Factors:\n")
    f.write(f"  min_alpha = {min_alpha}\n")
    f.write(f"  global_s = {global_s}\n")
    f.write(f"  alpha_scale = 2^{min_alpha} = {alpha_scale}\n")
    f.write(f"  full_scale = {full_scale}\n")
    f.write(f"  full_scale^2 = {full_scale_squared}\n")
    f.write("\n")

    f.write("Formula:\n")
    f.write("  var_real = var_hw * (full_scale)^2\n")
    f.write(f"  var_real = var_hw * {full_scale_squared}\n")
    f.write("\n")

    f.write("Statistics (var_real):\n")
    f.write(f"  Min:    {np.min(var_real_array):.8f}\n")
    f.write(f"  Max:    {np.max(var_real_array):.8f}\n")
    f.write(f"  Mean:   {np.mean(var_real_array):.8f}\n")
    f.write(f"  Median: {np.median(var_real_array):.8f}\n")
    f.write(f"  Std:    {np.std(var_real_array):.8f}\n")
    f.write("\n")

    f.write("All var_real values:\n")
    for i, var_real in enumerate(var_real_array):
        f.write(f"{i},{var_real:.10f}\n")

print(f"\nResults saved to: {output_file}")

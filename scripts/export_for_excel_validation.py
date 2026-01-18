"""
Export data to CSV for manual Excel validation
This allows you to verify the var_hw calculation step-by-step
"""

import os
import csv

print("="*80)
print("EXPORTING DATA FOR EXCEL VALIDATION")
print("="*80)

# Choose which vector to export (let's use vector 0 as example)
VECTOR_ID = 0

data_folder = "data/quantized"

# 1. Load global parameters
print(f"\n1. Loading global parameters...")
with open(os.path.join(data_folder, "global_params.txt"), "r") as f:
    lines = f.readlines()
    num_channels = int(lines[1].strip())
    num_vectors = int(lines[2].strip())
    global_s = float(lines[3].strip())
    global_zp = int(lines[4].strip())

print(f"   num_channels = {num_channels}")
print(f"   global_s = {global_s}")
print(f"   global_zp = {global_zp}")

# 2. Load alpha factors
print(f"\n2. Loading alpha factors...")
with open(os.path.join(data_folder, "alpha_factors.txt"), "r") as f:
    lines = f.readlines()
    alpha_factors = []
    for i in range(2, 2 + num_channels):
        alpha_factors.append(int(lines[i].strip()))

min_alpha = min(alpha_factors)
print(f"   min_alpha = {min_alpha}")

# 3. Load vector data
print(f"\n3. Loading vector {VECTOR_ID}...")
vec_file = os.path.join(data_folder, f"vector_{VECTOR_ID:03d}.txt")
with open(vec_file, "r") as f:
    lines = f.readlines()
    quantized_values = []
    for i in range(2, 2 + num_channels):
        quantized_values.append(int(lines[i].strip()))

print(f"   Loaded {len(quantized_values)} values")

# 4. Create Excel-friendly CSV
output_file = "data/lut/excel_validation.csv"
print(f"\n4. Creating CSV file: {output_file}")

with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)

    # Write header
    writer.writerow([
        'Channel_i',
        'q[i]',
        'zp',
        'alpha[i]',
        'min_alpha',
        'xi',
        '|xi|',
        'compressed',
        'shift',
        'xc_sq',
        'relative_shift',
        'Ex_contribution',
        'Ex2_contribution'
    ])

    # SQUARE_LUT
    SQUARE_LUT = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]

    # Calculate for each channel
    for i in range(num_channels):
        q = quantized_values[i]
        zp = global_zp
        alpha = alpha_factors[i]

        # 1. Center
        xi = q - zp

        # 2. Compress & Square
        abs_xi = abs(xi)
        if abs_xi >= 64:
            compressed = abs_xi >> 4
            shift = 1
        else:
            compressed = abs_xi >> 2
            shift = 0
        if compressed > 15:
            compressed = 15

        xc_sq = SQUARE_LUT[compressed] << (4 * shift)

        # 3. Scaling
        relative_shift = alpha - min_alpha
        ex_contribution = xi << relative_shift
        ex2_contribution = xc_sq << (2 * relative_shift)

        # Write row
        writer.writerow([
            i,
            q,
            zp,
            alpha,
            min_alpha,
            xi,
            abs_xi,
            compressed,
            shift,
            xc_sq,
            relative_shift,
            ex_contribution,
            ex2_contribution
        ])

print(f"\n5. CSV created successfully!")
print(f"\nNow open '{output_file}' in Excel and:")
print("   1. Sum column 'Ex_contribution' -> this is Ex")
print("   2. Sum column 'Ex2_contribution' -> this is Ex2")
print("   3. Calculate: mean_hw = Ex / 384")
print("   4. Calculate: mean_sq_hw = (Ex2 << 4) / 384")
print("   5. Calculate: var_hw = mean_sq_hw - mean_hw^2")

# Also save the expected results for comparison
print(f"\n6. Expected results for vector {VECTOR_ID}:")

# Calculate the correct answer
Ex = sum(quantized_values[i] - global_zp for i in range(num_channels))  # Simplified, actual needs shifts
# Actually calculate correctly
Ex = 0
Ex2 = 0
for i in range(num_channels):
    q = quantized_values[i]
    zp = global_zp
    alpha = alpha_factors[i]

    xi = q - zp
    abs_xi = abs(xi)

    if abs_xi >= 64:
        compressed = abs_xi >> 4
        shift = 1
    else:
        compressed = abs_xi >> 2
        shift = 0
    if compressed > 15:
        compressed = 15

    xc_sq = SQUARE_LUT[compressed] << (4 * shift)
    relative_shift = alpha - min_alpha

    Ex += xi << relative_shift
    Ex2 += xc_sq << (2 * relative_shift)

mean_hw = float(Ex) / num_channels
mean_sq_hw = float(Ex2 << 4) / num_channels
var_hw = mean_sq_hw - (mean_hw * mean_hw)

print(f"   Ex = {Ex}")
print(f"   Ex2 = {Ex2}")
print(f"   mean_hw = {mean_hw:.6f}")
print(f"   mean_sq_hw = {mean_sq_hw:.6f}")
print(f"   var_hw = {var_hw:.6f}")

# Save expected results to a separate file
with open("data/lut/expected_results.txt", 'w') as f:
    f.write(f"Expected Results for Vector {VECTOR_ID}\n")
    f.write("="*80 + "\n\n")
    f.write(f"Ex = {Ex}\n")
    f.write(f"Ex2 = {Ex2}\n")
    f.write(f"mean_hw = {mean_hw:.10f}\n")
    f.write(f"mean_sq_hw = {mean_sq_hw:.10f}\n")
    f.write(f"var_hw = {var_hw:.10f}\n")
    f.write(f"\nFormulas:\n")
    f.write(f"  mean_hw = Ex / {num_channels}\n")
    f.write(f"  mean_sq_hw = (Ex2 << 4) / {num_channels}  = (Ex2 * 16) / {num_channels}\n")
    f.write(f"  var_hw = mean_sq_hw - mean_hw^2\n")

print(f"\nExpected results saved to: data/lut/expected_results.txt")
print("="*80)

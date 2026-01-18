"""
Create a simple CSV to compare var_hw vs var_real in Excel
"""

import csv

print("="*80)
print("Creating simple comparison CSV for Excel")
print("="*80)

# 1. Load var_hw values
print("\n1. Loading var_hw values...")
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
                vec_id = int(parts[0])
                var_hw = float(parts[1])
                var_hw_list.append((vec_id, var_hw))

print(f"   Loaded {len(var_hw_list)} var_hw values")

# 2. Load var_real values from validation_report.txt
print("\n2. Loading var_real values from validation_report.txt...")
var_real_list = []
with open("validation/validation_report.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("|") and not line.startswith("|---"):
            # Skip headers
            if "Vec ID" in line or "Variance" in line:
                continue
            if "=" in line:
                continue

            parts = line.split("|")
            if len(parts) >= 5:
                try:
                    vec_id = int(parts[1].strip())
                    var_real = float(parts[4].strip())
                    var_real_list.append((vec_id, var_real))
                except:
                    pass

print(f"   Loaded {len(var_real_list)} var_real values")

# 3. Get scaling factor
print("\n3. Loading scaling factors...")
import os
data_folder = "data/quantized"

with open(os.path.join(data_folder, "global_params.txt"), "r") as f:
    lines = f.readlines()
    global_s = float(lines[3].strip())

with open(os.path.join(data_folder, "alpha_factors.txt"), "r") as f:
    lines = f.readlines()
    num_channels = 384
    alpha_factors = []
    for i in range(2, 2 + num_channels):
        alpha_factors.append(int(lines[i].strip()))

min_alpha = min(alpha_factors)
alpha_scale = 2.0 ** min_alpha
full_scale = alpha_scale * global_s
full_scale_squared = full_scale ** 2

print(f"   min_alpha = {min_alpha}")
print(f"   global_s = {global_s}")
print(f"   full_scale^2 = {full_scale_squared:.10e}")

# 4. Create CSV
output_file = "data/lut/var_comparison.csv"
print(f"\n4. Creating CSV: {output_file}")

with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)

    # Header
    writer.writerow([
        'Vector_ID',
        'var_hw',
        'var_real_from_C',
        'var_real_calculated',
        'difference',
        'relative_error_%'
    ])

    # Data
    for i in range(len(var_hw_list)):
        vec_id_hw, var_hw = var_hw_list[i]
        vec_id_real, var_real_from_c = var_real_list[i]

        # Check IDs match
        assert vec_id_hw == vec_id_real == i, f"ID mismatch at {i}"

        # Calculate var_real from var_hw
        var_real_calculated = var_hw * full_scale_squared

        # Calculate difference
        diff = abs(var_real_from_c - var_real_calculated)
        rel_error = (diff / var_real_from_c) * 100 if var_real_from_c != 0 else 0

        writer.writerow([
            i,
            var_hw,
            var_real_from_c,
            var_real_calculated,
            diff,
            rel_error
        ])

print(f"\n5. CSV created successfully!")
print(f"\nNow open '{output_file}' in Excel and check:")
print("   - Column C (var_real_from_C) should match Column D (var_real_calculated)")
print("   - Column E (difference) should be very small (~1e-7 or less)")
print("   - Column F (relative_error_%) should be near 0%")
print()
print("To verify the formula yourself in Excel:")
print(f"   In cell D2, type: =B2*{full_scale_squared:.10e}")
print("   Then copy down to all rows")
print("   It should match column C!")
print("="*80)

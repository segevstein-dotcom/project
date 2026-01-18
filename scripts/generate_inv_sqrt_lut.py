import numpy as np
import os

# LUT Configuration (from analysis results)
LUT_SIZE = 256
INPUT_DIVISOR = 128  # divide var_hw by this to get index (2^7)
OUTPUT_SCALE = 16384  # multiply 1/sqrt(x) by this to get 8-bit value (2^14)

def generate_lut():
    """
    Generate inverse square root LUT
    LUT[i] = round((1 / sqrt(i / INPUT_DIVISOR)) * OUTPUT_SCALE)
    """
    print("="*80)
    print("GENERATING INVERSE SQUARE ROOT LUT")
    print("="*80)
    print(f"LUT Size: {LUT_SIZE} entries")
    print(f"INPUT_DIVISOR: {INPUT_DIVISOR} (2^{int(np.log2(INPUT_DIVISOR))})")
    print(f"OUTPUT_SCALE: {OUTPUT_SCALE} (2^{int(np.log2(OUTPUT_SCALE))})")
    print()

    lut = []

    print(f"{'Index':<8} {'Input_val':>16} {'1/sqrt(x)':>16} {'Scaled':>12} {'LUT[i]':>8}")
    print("-"*80)

    for i in range(LUT_SIZE):
        # Calculate the input value (var_hw) this index represents
        # Index = var_hw / INPUT_DIVISOR, so var_hw = Index * INPUT_DIVISOR
        input_val = i * INPUT_DIVISOR

        # Handle zero/near-zero case
        if i == 0 or input_val < 1e-9:
            # For zero variance, 1/sqrt is infinite
            # Saturate to max value
            lut_value = 255
        else:
            # Calculate 1/sqrt(input_val)
            inv_sqrt = 1.0 / np.sqrt(input_val)

            # Scale to 8-bit range
            scaled = inv_sqrt * OUTPUT_SCALE

            # Round and clamp to [0, 255]
            lut_value = int(np.round(scaled))
            lut_value = max(0, min(255, lut_value))

        lut.append(lut_value)

        # Print first 20 and last 10 entries
        if i < 20 or i >= LUT_SIZE - 10:
            if i > 0 and input_val >= 1e-9:
                inv_sqrt = 1.0 / np.sqrt(input_val)
                scaled = inv_sqrt * OUTPUT_SCALE
                print(f"{i:<8} {input_val:>16.2f} {inv_sqrt:>16.8f} {scaled:>12.2f} {lut_value:>8}")
            else:
                print(f"{i:<8} {input_val:>16.2f} {'INF':>16} {'SAT':>12} {lut_value:>8}")
        elif i == 20:
            print("...")

    return lut

def save_lut(lut, filename):
    """Save LUT to file"""
    output_dir = "data/lut"
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        f.write(f"# Inverse Square Root LUT\n")
        f.write(f"# Size: {LUT_SIZE} entries\n")
        f.write(f"# INPUT_DIVISOR: {INPUT_DIVISOR} (var_hw / {INPUT_DIVISOR} = index)\n")
        f.write(f"# OUTPUT_SCALE: {OUTPUT_SCALE} (LUT[i] / {OUTPUT_SCALE} = 1/sqrt)\n")
        f.write(f"# Formula: LUT[i] = round((1 / sqrt(i / {INPUT_DIVISOR})) * {OUTPUT_SCALE})\n")
        f.write(f"#\n")
        f.write(f"# Usage in C:\n")
        f.write(f"#   int index = (int)(var_hw / {INPUT_DIVISOR});\n")
        f.write(f"#   if (index > 255) index = 255;\n")
        f.write(f"#   uint8_t lut_val = INV_SQRT_LUT[index];\n")
        f.write(f"#   float result = (float)lut_val / {OUTPUT_SCALE};\n")
        f.write(f"#\n")
        f.write(f"{LUT_SIZE}\n")

        for i, value in enumerate(lut):
            f.write(f"{value}\n")

    print(f"\nLUT saved to: {filepath}")
    return filepath

def validate_lut(lut):
    """Validate LUT against expected range from analysis"""
    print("\n" + "="*80)
    print("LUT VALIDATION")
    print("="*80)

    # Load analysis results
    analysis_file = "data/lut/var_hw_analysis.txt"
    if os.path.exists(analysis_file):
        with open(analysis_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Min var_hw:"):
                    min_var_hw = float(line.split(':')[1].strip())
                elif line.startswith("Max var_hw:"):
                    max_var_hw = float(line.split(':')[1].strip())

        print(f"Expected var_hw range: [{min_var_hw:.2f}, {max_var_hw:.2f}]")

        # Calculate expected index range
        min_index = int(min_var_hw / INPUT_DIVISOR)
        max_index = int(max_var_hw / INPUT_DIVISOR)

        print(f"Expected LUT index range: [{min_index}, {max_index}]")
        print(f"LUT indices used: {max_index - min_index + 1} out of {LUT_SIZE}")
        print(f"Utilization: {((max_index - min_index + 1) / LUT_SIZE) * 100:.1f}%")

        # Check LUT values in expected range
        print(f"\nLUT values in expected range:")
        print(f"  LUT[{min_index}] = {lut[min_index]} (for var_hw = {min_var_hw:.2f})")
        print(f"  LUT[{max_index}] = {lut[max_index if max_index < LUT_SIZE else LUT_SIZE-1]} "
              f"(for var_hw = {max_var_hw:.2f})")

        # Calculate actual 1/sqrt values
        actual_min = lut[max_index if max_index < LUT_SIZE else LUT_SIZE-1] / OUTPUT_SCALE
        actual_max = lut[min_index] / OUTPUT_SCALE
        expected_min = 1.0 / np.sqrt(max_var_hw)
        expected_max = 1.0 / np.sqrt(min_var_hw)

        print(f"\n1/sqrt values:")
        print(f"  At min var_hw: expected = {expected_max:.8f}, LUT gives = {actual_max:.8f}, "
              f"error = {abs(expected_max - actual_max):.8f}")
        print(f"  At max var_hw: expected = {expected_min:.8f}, LUT gives = {actual_min:.8f}, "
              f"error = {abs(expected_min - actual_min):.8f}")

        print(f"\nValidation: {'PASS' if max_index < LUT_SIZE else 'WARNING: max index exceeds LUT size'}")
    else:
        print(f"Warning: {analysis_file} not found, skipping validation")

def main():
    # Generate LUT
    lut = generate_lut()

    # Save to file
    filepath = save_lut(lut, "lut_inv_sqrt.txt")

    # Validate
    validate_lut(lut)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Update utils.h with:")
    print(f"   #define INV_SQRT_INPUT_DIVISOR {INPUT_DIVISOR}")
    print(f"   #define INV_SQRT_OUTPUT_SCALE {OUTPUT_SCALE}")
    print(f"2. Update inv_sqrt_lut() in utils.c to use division instead of multiplication")
    print(f"3. Call load_inv_sqrt_lut() in main.c")
    print(f"4. Implement Stage 2 (LayerNorm) using the LUT")

if __name__ == "__main__":
    main()

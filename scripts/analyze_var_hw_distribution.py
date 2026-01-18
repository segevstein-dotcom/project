import numpy as np
import os

# Square LUT from C code
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

def main():
    print("="*80)
    print("ANALYZING var_hw DISTRIBUTION FOR LUT DESIGN")
    print("="*80)

    # Load data
    data_folder = "data/quantized"

    # 1. Load global parameters
    with open(os.path.join(data_folder, "global_params.txt"), "r") as f:
        lines = f.readlines()
        num_channels = int(lines[1].strip())
        num_vectors = int(lines[2].strip())
        global_s = float(lines[3].strip())
        global_zp = int(lines[4].strip())

    print(f"\nGlobal Parameters:")
    print(f"  Channels: {num_channels}")
    print(f"  Vectors: {num_vectors}")
    print(f"  Global S: {global_s}")
    print(f"  Global ZP: {global_zp}")

    # 2. Load alpha factors (same for all vectors)
    with open(os.path.join(data_folder, "alpha_factors.txt"), "r") as f:
        lines = f.readlines()
        alpha_count = int(lines[1].strip())
        alpha_factors = []
        for i in range(2, 2 + num_channels):
            alpha_factors.append(int(lines[i].strip()))

    print(f"\nAlpha factors loaded: {len(alpha_factors)}")
    print(f"  Min alpha: {min(alpha_factors)}")
    print(f"  Max alpha: {max(alpha_factors)}")
    print(f"  Unique values: {sorted(set(alpha_factors))}")

    # 3. Zero points (all same as global_zp)
    zero_points = [global_zp] * num_channels

    # 4. Process all vectors and collect var_hw values
    var_hw_list = []
    mean_hw_list = []
    min_alpha_list = []

    print(f"\nProcessing {num_vectors} vectors...")
    for vec_idx in range(num_vectors):
        # Load vector
        vec_file = os.path.join(data_folder, f"vector_{vec_idx:03d}.txt")
        if not os.path.exists(vec_file):
            print(f"Warning: {vec_file} not found, skipping...")
            continue

        with open(vec_file, "r") as f:
            lines = f.readlines()
            quantized_values = []
            for i in range(2, 2 + num_channels):  # Skip header
                quantized_values.append(int(lines[i].strip()))

        # Compute statistics (simulating C code exactly)
        Ex, Ex2, min_alpha = stage1_statistics(quantized_values, zero_points, alpha_factors)
        mean_hw, var_hw = compute_variance_hw(Ex, Ex2, num_channels)

        var_hw_list.append(var_hw)
        mean_hw_list.append(mean_hw)
        min_alpha_list.append(min_alpha)

        if vec_idx % 20 == 0:
            print(f"  Processed {vec_idx}/{num_vectors}")

    var_hw_array = np.array(var_hw_list)
    mean_hw_array = np.array(mean_hw_list)

    print(f"\nProcessed {len(var_hw_list)} vectors successfully")

    # 5. Statistical analysis
    print("\n" + "="*80)
    print("var_hw DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Min:        {np.min(var_hw_array):.2f}")
    print(f"Max:        {np.max(var_hw_array):.2f}")
    print(f"Mean:       {np.mean(var_hw_array):.2f}")
    print(f"Median:     {np.median(var_hw_array):.2f}")
    print(f"Std:        {np.std(var_hw_array):.2f}")
    print(f"\nPercentiles:")
    print(f"  1%:  {np.percentile(var_hw_array, 1):.2f}")
    print(f"  5%:  {np.percentile(var_hw_array, 5):.2f}")
    print(f"  25%: {np.percentile(var_hw_array, 25):.2f}")
    print(f"  50%: {np.percentile(var_hw_array, 50):.2f}")
    print(f"  75%: {np.percentile(var_hw_array, 75):.2f}")
    print(f"  95%: {np.percentile(var_hw_array, 95):.2f}")
    print(f"  99%: {np.percentile(var_hw_array, 99):.2f}")

    # 6. LUT parameter recommendations
    print("\n" + "="*80)
    print("LUT PARAMETER RECOMMENDATIONS")
    print("="*80)

    # Calculate 1/sqrt(var_hw) range
    inv_sqrt_values = 1.0 / np.sqrt(var_hw_array)

    print(f"\n1/sqrt(var_hw) range:")
    print(f"  Min: {np.min(inv_sqrt_values):.6f}")
    print(f"  Max: {np.max(inv_sqrt_values):.6f}")
    print(f"  Mean: {np.mean(inv_sqrt_values):.6f}")

    # Recommend INPUT_SCALE
    max_var_hw = np.max(var_hw_array)
    min_var_hw = np.min(var_hw_array)

    print(f"\nFor 256-entry LUT:")
    print(f"  var_hw range: [{min_var_hw:.2f}, {max_var_hw:.2f}]")

    # Try different input scales (DIVISION, since var_hw is large!)
    print(f"\nEvaluating different INPUT_DIVISOR options (address = var_hw / divisor):")
    for divisor in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        max_index = max_var_hw / divisor
        min_index = min_var_hw / divisor
        coverage = (max_index <= 255)
        utilization = (max_index / 255.0) * 100 if max_index > 0 else 0

        print(f"  INPUT_DIVISOR = {divisor:3d} (2^{int(np.log2(divisor))}): "
              f"index range [{min_index:6.1f}, {max_index:6.1f}], "
              f"utilization: {utilization:5.1f}%, fits: {coverage}")

    # Recommend OUTPUT_SCALE
    max_inv_sqrt = np.max(inv_sqrt_values)
    min_inv_sqrt = np.min(inv_sqrt_values)

    print(f"\nFor 8-bit LUT output (0-255):")
    print(f"  1/sqrt(var_hw) range: [{min_inv_sqrt:.6f}, {max_inv_sqrt:.6f}]")

    print(f"\nEvaluating different OUTPUT_SCALE options:")
    for output_scale in [16, 32, 64, 128, 256, 512, 1024]:
        max_output = max_inv_sqrt * output_scale
        min_output = min_inv_sqrt * output_scale
        fits = (max_output <= 255)
        utilization = (max_output / 255.0) * 100 if max_output > 0 else 0

        print(f"  OUTPUT_SCALE = {output_scale:4d}: output range [{min_output:6.1f}, {max_output:6.1f}], "
              f"utilization: {utilization:5.1f}%, fits: {fits}")

    # Find optimal scales
    print("\n" + "="*80)
    print("RECOMMENDED CONFIGURATION")
    print("="*80)

    # Input divisor: choose one that uses 70-90% of the 256 entries
    optimal_input_divisor = max_var_hw / 200  # Target ~200 for good utilization
    divisor_options = [32, 64, 128, 256]
    best_input_divisor = min(divisor_options, key=lambda x: abs(x - optimal_input_divisor))

    # Output scale: choose one that uses 70-90% of 255
    optimal_output_scale = 200 / max_inv_sqrt  # Target ~200 for good utilization
    output_scale_options = [16384, 32768, 65536]  # Much larger scales needed!
    best_output_scale = min(output_scale_options, key=lambda x: abs(x - optimal_output_scale))

    print(f"INPUT_DIVISOR:  {best_input_divisor} (2^{int(np.log2(best_input_divisor))}) - divide var_hw to get index")
    print(f"OUTPUT_SCALE:   {best_output_scale} (2^{int(np.log2(best_output_scale))}) - multiply 1/sqrt to get 8-bit value")
    print(f"\nWith these parameters:")
    max_idx = max_var_hw / best_input_divisor
    min_idx = min_var_hw / best_input_divisor
    max_out = max_inv_sqrt * best_output_scale
    min_out = min_inv_sqrt * best_output_scale
    print(f"  LUT index range: [{min_idx:.1f}, {max_idx:.1f}] out of 255 ({(max_idx/255)*100:.1f}% utilization)")
    print(f"  LUT value range: [{min_out:.1f}, {max_out:.1f}] out of 255 ({(max_out/255)*100:.1f}% utilization)")

    # ============================================
    # REVERSE ENGINEERING: var_hw -> var_real
    # ============================================
    print("\n" + "="*80)
    print("REVERSE ENGINEERING: var_hw -> var_real")
    print("="*80)

    # Calculate scaling factors (from main.c lines 127-132)
    alpha_scale = 2.0 ** min_alpha
    full_scale = alpha_scale * global_s
    full_scale_squared = full_scale ** 2

    # Reverse transformation
    var_real_array = var_hw_array * full_scale_squared
    std_real_array = np.sqrt(var_real_array)

    print(f"\nScaling factors:")
    print(f"  min_alpha = {min_alpha}")
    print(f"  global_s = {global_s}")
    print(f"  alpha_scale = 2^{min_alpha} = {alpha_scale}")
    print(f"  full_scale = alpha_scale * global_s = {full_scale:.10f}")
    print(f"  full_scale^2 = {full_scale_squared:.10e}")

    print(f"\nFormula (from main.c lines 127-132):")
    print(f"  var_real = var_hw * (full_scale)^2")
    print(f"  var_real = var_hw * {full_scale_squared:.10e}")

    print(f"\nReal domain statistics:")
    print(f"  var_real range:  [{np.min(var_real_array):.6f}, {np.max(var_real_array):.6f}]")
    print(f"  var_real mean:   {np.mean(var_real_array):.6f}")
    print(f"  var_real median: {np.median(var_real_array):.6f}")
    print(f"  var_real std:    {np.std(var_real_array):.6f}")

    print(f"\n  std_real range:  [{np.min(std_real_array):.6f}, {np.max(std_real_array):.6f}]")
    print(f"  std_real mean:   {np.mean(std_real_array):.6f}")

    print(f"\nKey insight:")
    print(f"  var_hw is {int(1/full_scale_squared)}x LARGER than var_real")
    print(f"  This is due to 2 factors:")
    print(f"    1. min_alpha scaling: 2^(2*{min_alpha}) = {2**(2*min_alpha)}x")
    print(f"    2. global_s scaling:  (1/{global_s:.6f})^2 = {(1/global_s)**2:.2f}x")
    print(f"    Total: {int(1/full_scale_squared)}x")

    # Save results to file
    output_file = "data/lut/var_hw_analysis.txt"
    os.makedirs("data/lut", exist_ok=True)

    with open(output_file, "w") as f:
        f.write("var_hw Distribution Analysis\n")
        f.write("="*80 + "\n")
        f.write(f"Min var_hw: {np.min(var_hw_array):.6f}\n")
        f.write(f"Max var_hw: {np.max(var_hw_array):.6f}\n")
        f.write(f"Mean var_hw: {np.mean(var_hw_array):.6f}\n")
        f.write(f"Median var_hw: {np.median(var_hw_array):.6f}\n")
        f.write(f"\nRecommended LUT Parameters:\n")
        f.write(f"INPUT_DIVISOR: {best_input_divisor} (2^{int(np.log2(best_input_divisor))})\n")
        f.write(f"OUTPUT_SCALE: {best_output_scale} (2^{int(np.log2(best_output_scale))})\n")
        f.write(f"\nReverse Engineering (var_hw -> var_real):\n")
        f.write(f"Scaling factors:\n")
        f.write(f"  min_alpha = {min_alpha}\n")
        f.write(f"  global_s = {global_s}\n")
        f.write(f"  full_scale^2 = {full_scale_squared:.10e}\n")
        f.write(f"Real domain statistics:\n")
        f.write(f"  Min var_real: {np.min(var_real_array):.8f}\n")
        f.write(f"  Max var_real: {np.max(var_real_array):.8f}\n")
        f.write(f"  Mean var_real: {np.mean(var_real_array):.8f}\n")
        f.write(f"  Median var_real: {np.median(var_real_array):.8f}\n")
        f.write(f"  Mean std_real: {np.mean(std_real_array):.8f}\n")
        f.write(f"\nAll var_hw values:\n")
        for i, var_hw in enumerate(var_hw_list):
            f.write(f"{i},{var_hw:.6f}\n")

    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Detailed Computation Trace for One Vector
==========================================
Shows step-by-step calculation to understand where approximation errors come from.
"""

import numpy as np
import sys
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==========================================
# Configuration
# ==========================================
# Auto-detect if running from validation/ or implementation/ directory
SCRIPT_DIR = Path(__file__).parent
if SCRIPT_DIR.name == "validation":
    # Running from validation/ folder
    BASE_DIR = SCRIPT_DIR.parent
else:
    # Running from implementation/ folder
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data" / "quantized"
RAW_DIR = DATA_DIR / "raw_input_vectors"
GLOBAL_PARAMS_FILE = DATA_DIR / "global_params.txt"

# SQUARE LUT (same as C code)
SQUARE_LUT = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225]

# ==========================================
# Helper Functions (same as C code)
# ==========================================

def dynamic_compress(x):
    """
    Dynamic compression: 8-bit to 4-bit (same logic as C)
    Returns: (compressed, shift)
    """
    if x >= 64:
        compressed = x >> 4
        shift = 1
    else:
        compressed = x >> 2
        shift = 0

    if compressed > 15:
        compressed = 15

    return compressed, shift


def read_global_params():
    """Read global parameters"""
    with open(GLOBAL_PARAMS_FILE, 'r') as f:
        lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]

        hidden_dim = int(data_lines[0])
        num_vectors = int(data_lines[1])
        global_s = float(data_lines[2])
        global_zp = int(data_lines[3])

    return hidden_dim, num_vectors, global_s, global_zp


def read_raw_vector(vector_idx):
    """Read raw float vector (ground truth)"""
    file_path = RAW_DIR / f"raw_vec_{vector_idx:03d}.txt"

    with open(file_path, 'r') as f:
        lines = f.readlines()
        count = int(lines[0].strip())
        values = [float(line.strip()) for line in lines[1:count+1]]

    return np.array(values)


def read_quantized_vector(vector_idx):
    """Read quantized vector (uint8 values)"""
    file_path = DATA_DIR / f"vector_{vector_idx:03d}.txt"

    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
        count = int(data_lines[0])
        values = [int(data_lines[i+1]) for i in range(count)]

    return values


def read_alpha_factors():
    """Read alpha factors"""
    file_path = DATA_DIR / "alpha_factors.txt"

    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
        count = int(data_lines[0])
        values = [int(data_lines[i+1]) for i in range(count)]

    return values


# ==========================================
# Main Trace Function
# ==========================================

def trace_vector_computation(vec_idx, num_channels_to_show=10):
    """
    Perform detailed trace of vector computation

    Args:
        vec_idx: Vector index to trace
        num_channels_to_show: Number of channels to show in detail (default 10)
    """

    print("=" * 120)
    print(f"DETAILED COMPUTATION TRACE - Vector {vec_idx}")
    print("=" * 120)

    # Step 1: Load data
    print("\n[STEP 1: Loading Data]")
    hidden_dim, num_vectors, global_s, global_zp = read_global_params()
    raw_vector = read_raw_vector(vec_idx)
    quantized_vector = read_quantized_vector(vec_idx)
    alpha_factors = read_alpha_factors()

    print(f"  Hidden Dim:  {hidden_dim}")
    print(f"  Global S:    {global_s:.10f}")
    print(f"  Global ZP:   {global_zp}")
    print(f"  Num Channels: {len(quantized_vector)}")

    # Step 2: Find min_alpha
    print("\n[STEP 2: Finding Min Alpha]")
    min_alpha = min(alpha_factors)
    print(f"  Min Alpha: {min_alpha}")
    print(f"  Alpha range: [{min(alpha_factors)}, {max(alpha_factors)}]")

    # Step 3: Process each channel (show first N)
    print(f"\n[STEP 3: Channel-by-Channel Processing (showing first {num_channels_to_show} channels)]")
    print("-" * 120)
    print(f"{'Ch':<4} | {'Q_val':<5} | {'ZP':<5} | {'xi':<6} | {'|xi|':<5} | {'Comp':<4} | {'Sh':<2} | "
          f"{'xc^2':<8} | {'Alpha':<5} | {'Rel_Sh':<6} | {'Ex_Contrib':<12} | {'Ex2_Contrib':<15}")
    print("-" * 120)

    Ex = 0
    Ex2 = 0

    for i in range(min(num_channels_to_show, len(quantized_vector))):
        # Input values
        q_val = quantized_vector[i]
        zp = global_zp
        alpha = alpha_factors[i]

        # 1. Center
        xi = q_val - zp

        # 2. Compress & Square
        abs_xi = abs(xi)
        compressed, shift = dynamic_compress(abs_xi)
        xc_sq = SQUARE_LUT[compressed] << (4 * shift)

        # 3. Global Scale Logic (Shift Left Only)
        relative_shift = alpha - min_alpha  # Always >= 0

        ex_contribution = xi << relative_shift
        ex2_contribution = xc_sq << (2 * relative_shift)

        # 4. Accumulate
        Ex += ex_contribution
        Ex2 += ex2_contribution

        # Print row
        print(f"{i:<4} | {q_val:<5} | {zp:<5} | {xi:>6} | {abs_xi:<5} | {compressed:<4} | {shift:<2} | "
              f"{xc_sq:<8} | {alpha:>5} | {relative_shift:>6} | {ex_contribution:>12} | {ex2_contribution:>15}")

    if len(quantized_vector) > num_channels_to_show:
        print(f"  ... (processing remaining {len(quantized_vector) - num_channels_to_show} channels) ...")

        # Process remaining channels without printing
        for i in range(num_channels_to_show, len(quantized_vector)):
            q_val = quantized_vector[i]
            zp = global_zp
            alpha = alpha_factors[i]

            xi = q_val - zp
            abs_xi = abs(xi)
            compressed, shift = dynamic_compress(abs_xi)
            xc_sq = SQUARE_LUT[compressed] << (4 * shift)

            relative_shift = alpha - min_alpha
            ex_contribution = xi << relative_shift
            ex2_contribution = xc_sq << (2 * relative_shift)

            Ex += ex_contribution
            Ex2 += ex2_contribution

    print("-" * 120)

    # Step 4: Calculate statistics in HW domain
    print(f"\n[STEP 4: Accumulated Sums]")
    print(f"  Ex (sum):   {Ex}")
    print(f"  Ex2 (sum):  {Ex2}")

    print(f"\n[STEP 5: Statistics in HW Domain]")
    N = len(quantized_vector)
    mean_hw = Ex / N
    mean_sq_hw = (Ex2 << 4) / N  # << 4 because we used compressed squares
    var_hw = mean_sq_hw - (mean_hw * mean_hw)
    if var_hw < 0:
        var_hw = 0

    print(f"  Mean_HW:    {mean_hw:.6f}")
    print(f"  Mean_sq_HW: {mean_sq_hw:.6f}")
    print(f"  Var_HW:     {var_hw:.6f}")

    # Step 5: Convert to Real Domain
    print(f"\n[STEP 6: Scale Factors]")
    alpha_scale = 2.0 ** min_alpha
    full_scale = alpha_scale * global_s

    print(f"  Alpha Scale (2^min_alpha): {alpha_scale:.10f}")
    print(f"  Global S:                  {global_s:.10f}")
    print(f"  Full Scale (alpha * S):    {full_scale:.10f}")

    print(f"\n[STEP 7: Convert to Real Domain]")
    mean_real = mean_hw * full_scale
    var_real = var_hw * (full_scale * full_scale)  # Variance scales by s^2
    std_real = np.sqrt(var_real)

    print(f"  Mean_Real: {mean_real:.10f}")
    print(f"  Var_Real:  {var_real:.10f}")
    print(f"  Std_Real:  {std_real:.10f}")

    # Step 6: Compare with Ground Truth
    print(f"\n[STEP 8: Ground Truth Comparison]")
    py_mean = np.mean(raw_vector)
    py_var = np.var(raw_vector)
    py_std = np.std(raw_vector)

    print(f"  Python Mean: {py_mean:.10f}")
    print(f"  Python Var:  {py_var:.10f}")
    print(f"  Python Std:  {py_std:.10f}")

    print(f"\n[STEP 9: Error Analysis]")
    mean_error = abs(py_mean - mean_real)
    var_error = abs(py_var - var_real)
    std_error = abs(py_std - std_real)

    mean_error_pct = (mean_error / (abs(py_mean) + 1e-10)) * 100
    var_error_pct = (var_error / (abs(py_var) + 1e-10)) * 100
    std_error_pct = (std_error / (abs(py_std) + 1e-10)) * 100

    print(f"  Mean Error: {mean_error:.10f} ({mean_error_pct:.4f}%)")
    print(f"  Var Error:  {var_error:.10f} ({var_error_pct:.4f}%)")
    print(f"  Std Error:  {std_error:.10f} ({std_error_pct:.4f}%)")

    print("\n" + "=" * 120)
    print("SOURCES OF APPROXIMATION ERROR:")
    print("=" * 120)
    print("1. Dynamic Compression: 8-bit -> 4-bit lossy compression")
    print("   - Values >= 64: compressed = x >> 4 (loses 4 LSBs)")
    print("   - Values < 64:  compressed = x >> 2 (loses 2 LSBs)")
    print("")
    print("2. Square LUT: Only 16 entries (0-15)")
    print("   - Actual squares computed from compressed values, not original values")
    print("")
    print("3. Shift Operations: Used instead of exact multiplications")
    print("   - Left shifts are exact for powers of 2, but applied to approximate values")
    print("")
    print("4. Fixed-Point Arithmetic: Integer operations throughout")
    print("   - No floating-point precision until final conversion")
    print("=" * 120)


# ==========================================
# Main Entry Point
# ==========================================

def main():
    """Main trace workflow"""

    # Default: trace vector 0
    vec_idx = 0

    # Allow command-line argument
    if len(sys.argv) > 1:
        vec_idx = int(sys.argv[1])

    trace_vector_computation(vec_idx, num_channels_to_show=10)

    print(f"\nTo trace a different vector, run: python trace_vector.py <vector_index>")


if __name__ == "__main__":
    main()

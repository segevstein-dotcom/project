#!/usr/bin/env python3
"""
Compare C Implementation vs Dequantized Values
===============================================
This script compares the C SOLE algorithm results against values
reconstructed from quantization (dequantized values).

This shows the error from the ALGORITHM ONLY (dynamic compression, square LUT, etc.)
excluding the quantization error.

Compare with validate_statistics.py which shows TOTAL error (quantization + algorithm).
"""

import numpy as np
import os
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
C_REPORT_FILE = DATA_DIR / "final_report.txt"
GLOBAL_PARAMS_FILE = DATA_DIR / "global_params.txt"
OUTPUT_FILE = BASE_DIR / "validation" / "dequantized_comparison.txt"

# ==========================================
# File Reading Functions
# ==========================================

def read_global_params():
    """Read global parameters from global_params.txt"""
    with open(GLOBAL_PARAMS_FILE, 'r') as f:
        lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]

        hidden_dim = int(data_lines[0])
        num_vectors = int(data_lines[1])
        global_s = float(data_lines[2])
        global_zp = int(data_lines[3])

    return hidden_dim, num_vectors, global_s, global_zp


def read_quantized_vector(vector_idx):
    """Read quantized vector (uint8 values)"""
    file_path = DATA_DIR / f"vector_{vector_idx:03d}.txt"

    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
        count = int(data_lines[0])
        values = [int(data_lines[i+1]) for i in range(count)]

    return np.array(values, dtype=np.uint8)


def read_alpha_factors():
    """Read alpha factors"""
    file_path = DATA_DIR / "alpha_factors.txt"

    with open(file_path, 'r') as f:
        lines = f.readlines()
        data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
        count = int(data_lines[0])
        values = [int(data_lines[i+1]) for i in range(count)]

    return np.array(values, dtype=np.int8)


def read_c_results(report_file):
    """Parse C implementation results from final_report.txt"""
    results = []

    with open(report_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the table start
    table_started = False
    for line in lines:
        if '|' in line and 'Vec ID' in line:
            table_started = True
            continue

        # Skip separator lines
        if table_started and '|---' in line:
            continue

        # Parse data lines
        if table_started and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]

            if len(parts) >= 8:
                try:
                    vec_id = int(parts[1])
                    min_alpha = int(parts[2])
                    mean_real = float(parts[3])
                    var_real = float(parts[4])
                    std_real = float(parts[5])
                    ex = int(parts[6])

                    results.append({
                        'vec_id': vec_id,
                        'min_alpha': min_alpha,
                        'mean': mean_real,
                        'variance': var_real,
                        'std': std_real,
                        'ex': ex
                    })
                except (ValueError, IndexError):
                    continue

    return results


def dequantize_and_compute_stats(quantized_vector, alpha_factors, global_s, global_zp):
    """
    Dequantize the vector and compute statistics

    Args:
        quantized_vector: uint8 quantized values
        alpha_factors: per-channel alpha factors
        global_s: global scale factor
        global_zp: global zero point

    Returns:
        tuple: (mean, variance, std) of dequantized values
    """
    # Dequantize: x_real = (q - zp) * global_s * 2^alpha
    dequantized = np.zeros_like(quantized_vector, dtype=np.float32)

    for i in range(len(quantized_vector)):
        q = quantized_vector[i]
        alpha = alpha_factors[i]

        # Dequantize
        x_centered = float(q) - float(global_zp)
        scale = global_s * (2.0 ** alpha)
        dequantized[i] = x_centered * scale

    # Compute statistics
    mean = np.mean(dequantized)
    var = np.var(dequantized)
    std = np.std(dequantized)

    return mean, var, std


# ==========================================
# Comparison Logic
# ==========================================

def compare_c_vs_dequantized(num_vectors, global_s, global_zp):
    """
    Compare C implementation vs dequantized values

    This shows algorithm error only (excluding quantization error)
    """

    print("=" * 140)
    print(f"{'C Implementation vs Dequantized Values (Algorithm Error Only)':^140}")
    print("=" * 140)
    print(f"\n📊 Global Scale Factor (S): {global_s:.10f}")
    print(f"📦 Number of vectors: {num_vectors}\n")

    # Read data
    print(f"📖 Reading C results from: {C_REPORT_FILE}")
    c_results = read_c_results(C_REPORT_FILE)
    print(f"✓ Found {len(c_results)} vectors in C report\n")

    print(f"📖 Reading quantized data...")
    alpha_factors = read_alpha_factors()
    print(f"✓ Loaded alpha factors\n")

    # Open output file
    print(f"📝 Creating comparison report: {OUTPUT_FILE}\n")
    report_file = open(OUTPUT_FILE, 'w', encoding='utf-8')

    # Write file header
    report_file.write("=" * 160 + "\n")
    report_file.write(f"{'C IMPLEMENTATION vs DEQUANTIZED VALUES (Algorithm Error Only)':^160}\n")
    report_file.write("=" * 160 + "\n")
    report_file.write(f"This comparison shows error from the SOLE algorithm approximations ONLY.\n")
    report_file.write(f"Quantization error is excluded (both sides use quantized data).\n")
    report_file.write(f"\nGlobal Scale Factor (S): {global_s:.10f}\n")
    report_file.write(f"Number of vectors: {num_vectors}\n")
    report_file.write("=" * 160 + "\n\n")

    # Write table header
    header = f"{'Vec':<4} | {'Deq Mean':<14} | {'C Mean':<14} | {'Mean Err %':<11} | " \
             f"{'Deq Var':<14} | {'C Var':<14} | {'Var Err %':<11} | " \
             f"{'Deq Std':<14} | {'C Std':<14} | {'Std Err %':<11}"
    report_file.write(header + "\n")
    report_file.write("-" * 160 + "\n")

    print(header)
    print("-" * 140)

    # Store errors for summary
    errors = {
        'mean': [],
        'variance': [],
        'std': []
    }

    # Process each vector
    for c_result in c_results:
        vec_id = c_result['vec_id']

        try:
            # Read quantized vector
            quantized_vector = read_quantized_vector(vec_id)

            # Dequantize and compute statistics
            deq_mean, deq_var, deq_std = dequantize_and_compute_stats(
                quantized_vector, alpha_factors, global_s, global_zp
            )

            # C results
            c_mean = c_result['mean']
            c_var = c_result['variance']
            c_std = c_result['std']

            # Calculate percentage errors
            mean_error_pct = abs(deq_mean - c_mean) / (abs(deq_mean) + 1e-10) * 100
            var_error_pct = abs(deq_var - c_var) / (abs(deq_var) + 1e-10) * 100
            std_error_pct = abs(deq_std - c_std) / (abs(deq_std) + 1e-10) * 100

            # Store errors
            errors['mean'].append(mean_error_pct)
            errors['variance'].append(var_error_pct)
            errors['std'].append(std_error_pct)

            # Write ALL vectors to file
            row = f"{vec_id:<4} | {deq_mean:>14.6f} | {c_mean:>14.6f} | {mean_error_pct:>10.4f}% | " \
                  f"{deq_var:>14.6f} | {c_var:>14.6f} | {var_error_pct:>10.4f}% | " \
                  f"{deq_std:>14.6f} | {c_std:>14.6f} | {std_error_pct:>10.4f}%"
            report_file.write(row + "\n")

            # Print subsample to console
            if vec_id % 10 == 0 or vec_id < 5 or vec_id >= num_vectors - 5:
                print(row)

        except FileNotFoundError as e:
            error_msg = f"Vector {vec_id}: {e}"
            print(f"⚠️  {error_msg}")
            report_file.write(f"# ERROR: {error_msg}\n")
            continue

    report_file.write("=" * 160 + "\n\n")
    print("=" * 140)

    # Summary Statistics
    summary_header = "SUMMARY STATISTICS (Algorithm Error Only)"
    print(f"\n{summary_header:^140}")
    report_file.write(f"\n{summary_header:^160}\n")

    print("=" * 140)
    report_file.write("=" * 160 + "\n")

    header = f"{'Metric':<15} | {'Mean Error %':<15} | {'Max Error %':<15} | {'Min Error %':<15} | {'Std Dev %':<15}"
    print(header)
    report_file.write(header + "\n")

    print("-" * 140)
    report_file.write("-" * 160 + "\n")

    for metric_name in ['mean', 'variance', 'std']:
        metric_errors = errors[metric_name]

        mean_err = np.mean(metric_errors)
        max_err = np.max(metric_errors)
        min_err = np.min(metric_errors)
        std_err = np.std(metric_errors)

        row = f"{metric_name.capitalize():<15} | {mean_err:>14.6f}% | {max_err:>14.6f}% | " \
              f"{min_err:>14.6f}% | {std_err:>14.6f}%"
        print(row)
        report_file.write(row + "\n")

    print("=" * 140)
    report_file.write("=" * 160 + "\n\n")

    # Interpretation
    note = "INTERPRETATION:\n" \
           "===============\n" \
           "These errors show ONLY the approximation from the SOLE algorithm:\n" \
           "  - Dynamic compression (8-bit → 4-bit)\n" \
           "  - Square LUT (16 entries)\n" \
           "  - Fixed-point arithmetic\n" \
           "\n" \
           "Quantization error is NOT included here (both sides use quantized data).\n" \
           "\n" \
           "Compare with validation_report.txt to see TOTAL error (quantization + algorithm)."

    print(f"\n{note}\n")
    report_file.write(f"\n{note}\n")

    # Close file
    report_file.close()
    print(f"\n✅ Comparison report saved to: {OUTPUT_FILE}\n")


# ==========================================
# Main Entry Point
# ==========================================

def main():
    """Main comparison workflow"""

    print("\n🔍 Starting C vs Dequantized Comparison...\n")

    # Read global parameters
    print("📖 Reading global parameters...")
    hidden_dim, num_vectors, global_s, global_zp = read_global_params()
    print(f"✓ Hidden dim: {hidden_dim}, Vectors: {num_vectors}, Global S: {global_s:.8f}, Global ZP: {global_zp}\n")

    # Run comparison
    compare_c_vs_dequantized(num_vectors, global_s, global_zp)

    print("✅ Comparison complete!\n")


if __name__ == "__main__":
    main()

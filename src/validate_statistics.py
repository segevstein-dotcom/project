#!/usr/bin/env python3
"""
Validation Script: Compare C Implementation vs Ground Truth
============================================================
This script compares the statistics (mean, variance, std) computed by the C code
against the ground truth calculated from raw input vectors.

The C code works in HW domain and needs TWO corrections:
1. Alpha scale: 2^min_alpha
2. Global scale: global_s (THIS IS WHAT WE FIX HERE!)
"""

import numpy as np
import os
from pathlib import Path

# ==========================================
# Configuration
# ==========================================
# All paths relative to implementation/ directory
DATA_DIR = Path("data/quantized")
RAW_DIR = DATA_DIR / "raw_input_vectors"
REPORT_FILE = DATA_DIR / "final_report.txt"
GLOBAL_PARAMS_FILE = DATA_DIR / "global_params.txt"

# Verify paths exist (will be created by collect_real_data.py)
if not DATA_DIR.exists():
    print(f"⚠️  Error: Data directory not found: {DATA_DIR.absolute()}")
    print(f"   Please run 'python collect_real_data.py' first!")
    exit(1)

# ==========================================
# File Reading Functions
# ==========================================

def read_global_params():
    """
    Read global parameters from global_params.txt

    Returns:
        tuple: (hidden_dim, num_vectors, global_s, global_zp)
    """
    with open(GLOBAL_PARAMS_FILE, 'r') as f:
        lines = f.readlines()
        # Filter out comment lines
        data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]

        hidden_dim = int(data_lines[0])
        num_vectors = int(data_lines[1])
        global_s = float(data_lines[2])
        global_zp = int(data_lines[3])

    return hidden_dim, num_vectors, global_s, global_zp


def read_raw_vector(vector_idx):
    """
    Read raw float vector (before quantization)

    Args:
        vector_idx: Vector index (0-196)

    Returns:
        np.array: Raw vector values
    """
    file_path = RAW_DIR / f"raw_vec_{vector_idx:03d}.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Raw vector file not found: {file_path}")

    with open(file_path, 'r') as f:
        lines = f.readlines()
        count = int(lines[0].strip())
        values = [float(line.strip()) for line in lines[1:count+1]]

    return np.array(values)


def read_c_results(report_file):
    """
    Parse C implementation results from final_report.txt

    Returns:
        list of dict: Each dict contains vec_id, min_alpha, mean, variance, std, ex
    """
    if not os.path.exists(report_file):
        raise FileNotFoundError(f"C report file not found: {report_file}")

    results = []

    with open(report_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the table start (look for header with "Vec ID")
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
            # Split by '|' and clean up
            parts = [p.strip() for p in line.split('|')]

            if len(parts) >= 8:  # Should have at least 8 parts (including empty first/last)
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
                    # Skip malformed lines
                    continue

    return results


# ==========================================
# Validation Logic
# ==========================================

def compare_statistics(num_vectors, global_s):
    """
    Compare Python ground truth vs C implementation

    Args:
        num_vectors: Total number of vectors
        global_s: Global scale factor from quantization
    """

    print("=" * 140)
    print(f"{'VALIDATION: Python (Ground Truth) vs C Implementation':^140}")
    print("=" * 140)
    print(f"\n📊 Global Scale Factor (S): {global_s:.10f}")
    print(f"📦 Number of vectors: {num_vectors}\n")

    # Read C results
    print(f"📖 Reading C results from: {REPORT_FILE}")
    c_results = read_c_results(REPORT_FILE)
    print(f"✓ Found {len(c_results)} vectors in C report\n")

    # Table Header
    print(f"{'Vec':<4} | {'Python Mean':<14} | {'C Mean*':<14} | {'Mean Err %':<11} | "
          f"{'Python Var':<14} | {'C Var*':<14} | {'Var Err %':<11} | "
          f"{'Python Std':<14} | {'C Std*':<14} | {'Std Err %':<11}")
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
            # Read raw vector (ground truth)
            raw_vector = read_raw_vector(vec_id)

            # Calculate Python statistics (GROUND TRUTH)
            py_mean = np.mean(raw_vector)
            py_var = np.var(raw_vector)
            py_std = np.std(raw_vector)

            # C results (partial - already scaled by 2^min_alpha, but MISSING global_s)
            c_mean_partial = c_result['mean']
            c_var_partial = c_result['variance']
            c_std_partial = c_result['std']

            # ✅ FIX: Apply global_s correction
            c_mean_fixed = c_mean_partial * global_s
            c_var_fixed = c_var_partial * (global_s ** 2)  # Var(aX) = a² Var(X)
            c_std_fixed = c_std_partial * global_s

            # Calculate percentage errors
            mean_error_pct = abs(py_mean - c_mean_fixed) / (abs(py_mean) + 1e-10) * 100
            var_error_pct = abs(py_var - c_var_fixed) / (abs(py_var) + 1e-10) * 100
            std_error_pct = abs(py_std - c_std_fixed) / (abs(py_std) + 1e-10) * 100

            # Store errors
            errors['mean'].append(mean_error_pct)
            errors['variance'].append(var_error_pct)
            errors['std'].append(std_error_pct)

            # Print every vector (or subsample for large datasets)
            if vec_id % 10 == 0 or vec_id < 5 or vec_id >= num_vectors - 5:
                print(f"{vec_id:<4} | {py_mean:>14.6f} | {c_mean_fixed:>14.6f} | {mean_error_pct:>10.4f}% | "
                      f"{py_var:>14.6f} | {c_var_fixed:>14.6f} | {var_error_pct:>10.4f}% | "
                      f"{py_std:>14.6f} | {c_std_fixed:>14.6f} | {std_error_pct:>10.4f}%")

        except FileNotFoundError as e:
            print(f"⚠️  Vector {vec_id}: {e}")
            continue

    print("=" * 140)

    # Summary Statistics
    print(f"\n{'📈 SUMMARY STATISTICS':^140}")
    print("=" * 140)
    print(f"{'Metric':<15} | {'Mean Error %':<15} | {'Max Error %':<15} | {'Min Error %':<15} | {'Std Dev %':<15}")
    print("-" * 140)

    for metric_name in ['mean', 'variance', 'std']:
        metric_errors = errors[metric_name]

        mean_err = np.mean(metric_errors)
        max_err = np.max(metric_errors)
        min_err = np.min(metric_errors)
        std_err = np.std(metric_errors)

        print(f"{metric_name.capitalize():<15} | {mean_err:>14.6f}% | {max_err:>14.6f}% | "
              f"{min_err:>14.6f}% | {std_err:>14.6f}%")

    print("=" * 140)

    # Pass/Fail Criteria
    mean_avg_error = np.mean(errors['mean'])
    var_avg_error = np.mean(errors['variance'])
    std_avg_error = np.mean(errors['std'])

    print(f"\n{'🎯 VALIDATION RESULT':^140}")
    print("=" * 140)

    THRESHOLD = 5.0  # 5% error threshold

    all_passed = (mean_avg_error < THRESHOLD and
                  var_avg_error < THRESHOLD and
                  std_avg_error < THRESHOLD)

    if all_passed:
        print(f"✅ PASSED: All metrics within {THRESHOLD}% error threshold")
        print(f"   - Mean error:     {mean_avg_error:.4f}%")
        print(f"   - Variance error: {var_avg_error:.4f}%")
        print(f"   - Std error:      {std_avg_error:.4f}%")
    else:
        print(f"❌ FAILED: Some metrics exceed {THRESHOLD}% error threshold")
        if mean_avg_error >= THRESHOLD:
            print(f"   ⚠️  Mean error: {mean_avg_error:.4f}% (threshold: {THRESHOLD}%)")
        if var_avg_error >= THRESHOLD:
            print(f"   ⚠️  Variance error: {var_avg_error:.4f}% (threshold: {THRESHOLD}%)")
        if std_avg_error >= THRESHOLD:
            print(f"   ⚠️  Std error: {std_avg_error:.4f}% (threshold: {THRESHOLD}%)")

    print("=" * 140)
    print("\n* C Mean/Var/Std are CORRECTED by multiplying with global_s")
    print(f"  Formula: C_corrected = C_raw × global_s (for mean/std) or C_raw × global_s² (for variance)\n")


# ==========================================
# Main Entry Point
# ==========================================

def main():
    """Main validation workflow"""

    print("\n🔍 Starting Statistics Validation...\n")

    # Step 1: Read global parameters
    print("📖 Reading global parameters...")
    hidden_dim, num_vectors, global_s, global_zp = read_global_params()
    print(f"✓ Hidden dim: {hidden_dim}, Vectors: {num_vectors}, Global S: {global_s:.8f}, Global ZP: {global_zp}\n")

    # Step 2: Run comparison
    compare_statistics(num_vectors, global_s)

    print("✅ Validation complete!\n")


if __name__ == "__main__":
    main()

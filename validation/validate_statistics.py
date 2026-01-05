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
REPORT_FILE = DATA_DIR / "final_report.txt"
VALIDATION_REPORT_FILE = BASE_DIR / "validation" / "validation_report.txt"
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

    # Open file for detailed report
    print(f"📝 Creating detailed validation report: {VALIDATION_REPORT_FILE}\n")
    report_file = open(VALIDATION_REPORT_FILE, 'w', encoding='utf-8')

    # Write file header
    report_file.write("=" * 160 + "\n")
    report_file.write(f"{'VALIDATION REPORT: C Implementation vs Golden Reference':^160}\n")
    report_file.write("=" * 160 + "\n")
    report_file.write(f"Global Scale Factor (S): {global_s:.10f}\n")
    report_file.write(f"Number of vectors: {num_vectors}\n")
    report_file.write(f"Date: {os.popen('date /t').read().strip() if sys.platform == 'win32' else os.popen('date').read().strip()}\n")
    report_file.write("=" * 160 + "\n\n")

    # Write table header to file
    report_file.write(f"{'Vec':<4} | {'Golden Mean':<14} | {'C Mean':<14} | {'Mean Err %':<11} | "
                     f"{'Golden Var':<14} | {'C Var':<14} | {'Var Err %':<11} | "
                     f"{'Golden Std':<14} | {'C Std':<14} | {'Std Err %':<11}\n")
    report_file.write("-" * 160 + "\n")

    # Console table header
    print(f"{'Vec':<4} | {'Golden Mean':<14} | {'C Mean':<14} | {'Mean Err %':<11} | "
          f"{'Golden Var':<14} | {'C Var':<14} | {'Var Err %':<11} | "
          f"{'Golden Std':<14} | {'C Std':<14} | {'Std Err %':<11}")
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

            # Calculate Python statistics (GROUND TRUTH / GOLDEN REFERENCE)
            py_mean = np.mean(raw_vector)
            py_var = np.var(raw_vector)
            py_std = np.std(raw_vector)

            # C results (already in full Real domain - C code multiplies by 2^min_alpha AND global_s)
            c_mean = c_result['mean']
            c_var = c_result['variance']
            c_std = c_result['std']

            # Calculate percentage errors (no correction needed - C already outputs Real domain values)
            mean_error_pct = abs(py_mean - c_mean) / (abs(py_mean) + 1e-10) * 100
            var_error_pct = abs(py_var - c_var) / (abs(py_var) + 1e-10) * 100
            std_error_pct = abs(py_std - c_std) / (abs(py_std) + 1e-10) * 100

            # Store errors
            errors['mean'].append(mean_error_pct)
            errors['variance'].append(var_error_pct)
            errors['std'].append(std_error_pct)

            # Write ALL vectors to file
            report_file.write(f"{vec_id:<4} | {py_mean:>14.6f} | {c_mean:>14.6f} | {mean_error_pct:>10.4f}% | "
                            f"{py_var:>14.6f} | {c_var:>14.6f} | {var_error_pct:>10.4f}% | "
                            f"{py_std:>14.6f} | {c_std:>14.6f} | {std_error_pct:>10.4f}%\n")

            # Print subsample to console (every 10th, first 5, last 5)
            if vec_id % 10 == 0 or vec_id < 5 or vec_id >= num_vectors - 5:
                print(f"{vec_id:<4} | {py_mean:>14.6f} | {c_mean:>14.6f} | {mean_error_pct:>10.4f}% | "
                      f"{py_var:>14.6f} | {c_var:>14.6f} | {var_error_pct:>10.4f}% | "
                      f"{py_std:>14.6f} | {c_std:>14.6f} | {std_error_pct:>10.4f}%")

        except FileNotFoundError as e:
            error_msg = f"Vector {vec_id}: {e}"
            print(f"⚠️  {error_msg}")
            report_file.write(f"# ERROR: {error_msg}\n")
            continue

    report_file.write("=" * 160 + "\n\n")
    print("=" * 140)

    # Summary Statistics
    summary_header = "SUMMARY STATISTICS"
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

    # Pass/Fail Criteria
    mean_avg_error = np.mean(errors['mean'])
    var_avg_error = np.mean(errors['variance'])
    std_avg_error = np.mean(errors['std'])

    result_header = "VALIDATION RESULT"
    print(f"\n{result_header:^140}")
    report_file.write(f"\n{result_header:^160}\n")

    print("=" * 140)
    report_file.write("=" * 160 + "\n")

    THRESHOLD = 5.0  # 5% error threshold

    all_passed = (mean_avg_error < THRESHOLD and
                  var_avg_error < THRESHOLD and
                  std_avg_error < THRESHOLD)

    if all_passed:
        result_msg = f"PASSED: All metrics within {THRESHOLD}% error threshold"
        print(f"✅ {result_msg}")
        report_file.write(f"RESULT: {result_msg}\n")

        details = [
            f"   - Mean error:     {mean_avg_error:.4f}%",
            f"   - Variance error: {var_avg_error:.4f}%",
            f"   - Std error:      {std_avg_error:.4f}%"
        ]
        for detail in details:
            print(detail)
            report_file.write(detail + "\n")
    else:
        result_msg = f"FAILED: Some metrics exceed {THRESHOLD}% error threshold"
        print(f"❌ {result_msg}")
        report_file.write(f"RESULT: {result_msg}\n")

        if mean_avg_error >= THRESHOLD:
            msg = f"   Mean error: {mean_avg_error:.4f}% (threshold: {THRESHOLD}%)"
            print(f"   ⚠️  {msg}")
            report_file.write(f"   WARNING: {msg}\n")
        if var_avg_error >= THRESHOLD:
            msg = f"   Variance error: {var_avg_error:.4f}% (threshold: {THRESHOLD}%)"
            print(f"   ⚠️  {msg}")
            report_file.write(f"   WARNING: {msg}\n")
        if std_avg_error >= THRESHOLD:
            msg = f"   Std error: {std_avg_error:.4f}% (threshold: {THRESHOLD}%)"
            print(f"   ⚠️  {msg}")
            report_file.write(f"   WARNING: {msg}\n")

    print("=" * 140)
    report_file.write("=" * 160 + "\n\n")

    note = "Note: C Mean/Var/Std are in full Real domain (already multiplied by 2^min_alpha AND global_s)\n" \
           "      No additional correction needed - direct comparison with Python ground truth."
    print(f"\n{note}\n")
    report_file.write(f"{note}\n")

    # Close file
    report_file.close()
    print(f"\n✅ Detailed validation report saved to: {VALIDATION_REPORT_FILE}\n")


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

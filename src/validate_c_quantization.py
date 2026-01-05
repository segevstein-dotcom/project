import numpy as np
import os
import sys

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==========================================
# Configuration
# ==========================================
QUANTIZED_DIR = "data/quantized"
OUTPUT_DIR = "data"

# ==========================================
# Read Quantized Data (as C code would)
# ==========================================

def read_global_params():
    """Read global quantization parameters"""
    with open(os.path.join(QUANTIZED_DIR, "global_params.txt"), 'r') as f:
        lines = f.readlines()
        hidden_dim = int(lines[1].strip())
        num_vectors = int(lines[2].strip())
        global_s = float(lines[3].strip())
        global_zp = int(lines[4].strip())

    return hidden_dim, num_vectors, global_s, global_zp


def read_alpha_factors():
    """Read per-channel alpha factors"""
    with open(os.path.join(QUANTIZED_DIR, "alpha_factors.txt"), 'r') as f:
        lines = f.readlines()
        num_channels = int(lines[1].strip())
        alphas = []
        for i in range(2, 2 + num_channels):
            alphas.append(int(lines[i].strip()))

    return np.array(alphas)


def read_quantized_vector(vector_id):
    """Read a single quantized vector (INT8)"""
    filename = os.path.join(QUANTIZED_DIR, f"vector_{vector_id:03d}.txt")
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_elements = int(lines[1].strip())
        values = []
        for i in range(2, 2 + num_elements):
            values.append(int(lines[i].strip()))

    return np.array(values, dtype=np.int32)


def read_raw_vector(vector_id):
    """Read original float vector for comparison"""
    filename = os.path.join(QUANTIZED_DIR, "raw_input_vectors", f"raw_vec_{vector_id:03d}.txt")
    with open(filename, 'r') as f:
        lines = f.readlines()
        num_elements = int(lines[0].strip())
        values = []
        for i in range(1, 1 + num_elements):
            values.append(float(lines[i].strip()))

    return np.array(values)


# ==========================================
# Dequantization (as C code would do)
# ==========================================

def dequantize_vector(quantized_vector, global_s, global_zp, alpha_factors):
    """
    Dequantize INT8 vector back to float (SOLE method)

    Sequential Dequantization:
    1. X_stretched = X_int - ZP
    2. X_norm = X_stretched × 2^α
    3. X_real = X_norm × S
    """
    reconstructed = np.zeros(len(quantized_vector), dtype=np.float32)

    for i in range(len(quantized_vector)):
        q_val = int(quantized_vector[i])
        zp = int(global_zp)
        alpha = int(alpha_factors[i])
        s = float(global_s)

        # Sequential Dequantization
        x_stretched = q_val - zp
        x_norm = x_stretched * (2 ** alpha)
        x_real = x_norm * s

        reconstructed[i] = x_real

    return reconstructed


# ==========================================
# Calculate Statistics
# ==========================================

def calculate_statistics(vector):
    """Calculate mean, variance, std for a vector"""
    mean = np.mean(vector)
    variance = np.var(vector)
    std = np.std(vector)
    return mean, variance, std


def calculate_error_percentage(original, reconstructed):
    """Calculate percentage error"""
    if abs(original) < 1e-9:
        return 0.0  # Avoid division by near-zero
    return abs(original - reconstructed) / abs(original) * 100


# ==========================================
# Main Validation
# ==========================================

def main():
    print("="*80)
    print("C CODE QUANTIZATION VALIDATION")
    print("="*80)
    print()

    # Read global parameters
    print("Reading quantization parameters...")
    hidden_dim, num_vectors, global_s, global_zp = read_global_params()
    alpha_factors = read_alpha_factors()

    print(f"  Vectors: {num_vectors}")
    print(f"  Channels: {hidden_dim}")
    print(f"  Global S: {global_s:.8f}")
    print(f"  Global ZP: {global_zp}")
    print()

    # Process all vectors
    per_vector_stats = []

    print("Processing vectors...")
    for vec_id in range(num_vectors):
        # Read original and quantized
        original_vec = read_raw_vector(vec_id)
        quantized_vec = read_quantized_vector(vec_id)

        # Dequantize (simulate C code)
        reconstructed_vec = dequantize_vector(quantized_vec, global_s, global_zp, alpha_factors)

        # Calculate statistics for original
        mean_orig, var_orig, std_orig = calculate_statistics(original_vec)

        # Calculate statistics for reconstructed (from C code simulation)
        mean_recon, var_recon, std_recon = calculate_statistics(reconstructed_vec)

        # Calculate errors
        mean_err_pct = calculate_error_percentage(mean_orig, mean_recon)
        var_err_pct = calculate_error_percentage(var_orig, var_recon)
        std_err_pct = calculate_error_percentage(std_orig, std_recon)

        per_vector_stats.append({
            'vec_id': vec_id,
            'mean_orig': mean_orig,
            'var_orig': var_orig,
            'std_orig': std_orig,
            'mean_recon': mean_recon,
            'var_recon': var_recon,
            'std_recon': std_recon,
            'mean_err_pct': mean_err_pct,
            'var_err_pct': var_err_pct,
            'std_err_pct': std_err_pct
        })

        if (vec_id + 1) % 50 == 0:
            print(f"  Processed {vec_id + 1}/{num_vectors} vectors...")

    print(f"✓ Processed all {num_vectors} vectors")
    print()

    # ==========================================
    # Save Per-Vector Statistics
    # ==========================================

    output_file = os.path.join(OUTPUT_DIR, "c_code_per_vector_validation.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*150 + "\n")
        f.write("C CODE QUANTIZATION VALIDATION - PER-VECTOR STATISTICS\n")
        f.write("="*150 + "\n")
        f.write(f"Method: SOLE (Symmetric Fixed ZP=128, No Clipping)\n")
        f.write(f"Global S: {global_s:.10f}\n")
        f.write(f"Global ZP: {global_zp}\n")
        f.write("="*150 + "\n\n")

        # Table header
        f.write(f"{'Vec':<5} | {'Mean_Orig':<12} | {'Var_Orig':<12} | {'Std_Orig':<12} | "
               f"{'Mean_Recon':<12} | {'Var_Recon':<12} | {'Std_Recon':<12} | "
               f"{'Mean_Err%':<10} | {'Var_Err%':<10} | {'Std_Err%':<10}\n")
        f.write("-"*150 + "\n")

        # Data rows
        for stat in per_vector_stats:
            f.write(f"{stat['vec_id']:<5} | "
                   f"{stat['mean_orig']:12.8f} | {stat['var_orig']:12.8f} | {stat['std_orig']:12.8f} | "
                   f"{stat['mean_recon']:12.8f} | {stat['var_recon']:12.8f} | {stat['std_recon']:12.8f} | "
                   f"{stat['mean_err_pct']:10.6f} | {stat['var_err_pct']:10.6f} | {stat['std_err_pct']:10.6f}\n")

        # Summary statistics
        f.write("\n" + "="*150 + "\n")
        f.write("SUMMARY STATISTICS (across all vectors)\n")
        f.write("="*150 + "\n")

        mean_errs = [s['mean_err_pct'] for s in per_vector_stats]
        var_errs = [s['var_err_pct'] for s in per_vector_stats]
        std_errs = [s['std_err_pct'] for s in per_vector_stats]

        f.write(f"Mean Error %:     Avg={np.mean(mean_errs):.6f}%  Max={np.max(mean_errs):.6f}%  Min={np.min(mean_errs):.6f}%\n")
        f.write(f"Variance Error %: Avg={np.mean(var_errs):.6f}%  Max={np.max(var_errs):.6f}%  Min={np.min(var_errs):.6f}%\n")
        f.write(f"Std Error %:      Avg={np.mean(std_errs):.6f}%  Max={np.max(std_errs):.6f}%  Min={np.min(std_errs):.6f}%\n")

    print(f"✓ Per-vector statistics saved to: {output_file}")

    # ==========================================
    # Save Summary
    # ==========================================

    summary_file = os.path.join(OUTPUT_DIR, "c_code_validation_summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("C CODE QUANTIZATION VALIDATION - SUMMARY\n")
        f.write("="*120 + "\n")
        f.write(f"Total Vectors: {num_vectors}\n")
        f.write(f"Channels per Vector: {hidden_dim}\n")
        f.write(f"Global S: {global_s:.10f}\n")
        f.write(f"Global ZP: {global_zp}\n")
        f.write("="*120 + "\n\n")

        f.write("AVERAGE ERRORS ACROSS ALL VECTORS\n")
        f.write("="*120 + "\n\n")

        f.write(f"{'Metric':<20} | {'Avg Error %':<15} | {'Max Error %':<15} | {'Min Error %':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Mean':<20} | {np.mean(mean_errs):>14.8f}% | {np.max(mean_errs):>14.8f}% | {np.min(mean_errs):>14.8f}%\n")
        f.write(f"{'Variance':<20} | {np.mean(var_errs):>14.8f}% | {np.max(var_errs):>14.8f}% | {np.min(var_errs):>14.8f}%\n")
        f.write(f"{'Std':<20} | {np.mean(std_errs):>14.8f}% | {np.max(std_errs):>14.8f}% | {np.min(std_errs):>14.8f}%\n")

        f.write("\n" + "="*120 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*120 + "\n\n")

        f.write(f"The C code quantization accuracy (SOLE method with ZP=128, No Clipping):\n")
        f.write(f"  - Mean Error:     {np.mean(mean_errs):.6f}% (average across all vectors)\n")
        f.write(f"  - Variance Error: {np.mean(var_errs):.6f}% (average across all vectors)\n")
        f.write(f"  - Std Error:      {np.mean(std_errs):.6f}% (average across all vectors)\n\n")

        if np.mean(mean_errs) < 10 and np.mean(var_errs) < 1 and np.mean(std_errs) < 1:
            f.write("✅ EXCELLENT: C code quantization is highly accurate!\n")
        elif np.mean(mean_errs) < 20 and np.mean(var_errs) < 2 and np.mean(std_errs) < 2:
            f.write("✅ GOOD: C code quantization maintains good accuracy.\n")
        else:
            f.write("⚠️  WARNING: C code quantization has significant errors.\n")

    print(f"✓ Summary saved to: {summary_file}")
    print()

    # Print summary to console
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print(f"Average Mean Error:     {np.mean(mean_errs):>8.4f}%")
    print(f"Average Variance Error: {np.mean(var_errs):>8.4f}%")
    print(f"Average Std Error:      {np.mean(std_errs):>8.4f}%")
    print("="*80)
    print()
    print("✓ Validation complete!")


if __name__ == "__main__":
    main()

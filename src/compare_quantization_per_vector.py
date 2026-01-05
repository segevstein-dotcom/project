import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import requests
import numpy as np
import math
import sys
import os

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = 'facebook/deit-small-patch16-224'
CLIP_THRESHOLD = 7.0
OUTPUT_DIR = "data"

# ==========================================
# Quantization Functions (SOLE + Alpha)
# ==========================================

def quantize_with_sole(vectors_matrix, zp_method='symmetric_fixed', use_clipping=True):
    """
    SOLE Quantization with Alpha Factors (per-channel stretching)
    """
    num_vectors, num_channels = vectors_matrix.shape
    data = vectors_matrix.copy()

    # Step 1: Outlier Clipping (Optional)
    if use_clipping:
        data = np.clip(data, -CLIP_THRESHOLD, CLIP_THRESHOLD)

    # Step 2: Calculate Global Scale (S) and Zero Point (ZP)
    global_min = float(np.min(data))
    global_max = float(np.max(data))
    max_abs_value = max(abs(global_min), abs(global_max))

    if zp_method == 'asymmetric':
        global_s = (global_max - global_min) / 255.0
        global_zp = round(-global_min / global_s) if global_s > 0 else 128
    elif zp_method == 'symmetric_fixed':
        global_s = max_abs_value / 127.0
        global_zp = 128
    elif zp_method == 'symmetric_corrected':
        global_s = max_abs_value / 127.0
        normalized_min = global_min / global_s
        normalized_max = global_max / global_s
        normalized_center = (normalized_min + normalized_max) / 2
        global_zp = round(128 - normalized_center)
    else:
        raise ValueError(f"Unknown zp_method: {zp_method}")

    # Step 3: Calculate Per-Channel Alpha Factors
    alpha_factors = []
    processed_matrix = np.zeros_like(data)

    for channel_idx in range(num_channels):
        raw_values = data[:, channel_idx]
        normalized_values = raw_values / global_s

        norm_min = float(np.min(normalized_values))
        norm_max = float(np.max(normalized_values))
        norm_max_abs = max(abs(norm_min), abs(norm_max))

        if norm_max_abs < 1e-9:
            optimal_alpha = 0
        else:
            max_stretch = 127.0 / norm_max_abs
            if max_stretch >= 1.0:
                abs_alpha = int(math.floor(math.log2(max_stretch)))
            else:
                abs_alpha = 0
            optimal_alpha = -abs_alpha

        alpha_factors.append(optimal_alpha)

        stretched_values = normalized_values / (2 ** optimal_alpha)
        quantized_values = stretched_values + global_zp

        processed_matrix[:, channel_idx] = quantized_values

    # Step 4: Quantize to INT8 [0-255]
    quantized_vectors = []
    for vector_idx in range(num_vectors):
        row_values = processed_matrix[vector_idx, :]
        quantized_vector = np.round(row_values).astype(np.int32)
        quantized_vector = np.clip(quantized_vector, 0, 255)
        quantized_vectors.append(quantized_vector)

    params = {
        'global_s': global_s,
        'global_zp': global_zp,
        'alpha_factors': alpha_factors,
        'zp_method': zp_method,
        'use_clipping': use_clipping
    }

    return quantized_vectors, params


def dequantize_with_sole(quantized_vectors, params):
    """
    SOLE Dequantization (INT8 → Float)
    """
    global_s = params['global_s']
    global_zp = params['global_zp']
    alpha_factors = params['alpha_factors']

    reconstructed_matrix = []

    for quantized_vector in quantized_vectors:
        reconstructed_values = []

        for i in range(len(quantized_vector)):
            q_val = int(quantized_vector[i])
            zp = int(global_zp)
            alpha = int(alpha_factors[i])
            s = float(global_s)

            # Sequential Dequantization
            x_stretched = q_val - zp
            x_norm = x_stretched * (2 ** alpha)
            x_real = x_norm * s

            reconstructed_values.append(x_real)

        reconstructed_matrix.append(reconstructed_values)

    return np.array(reconstructed_matrix)


def calculate_per_vector_statistics(original, reconstructed):
    """
    Calculate per-vector statistics: mean, variance, std for original and reconstructed
    """
    num_vectors = original.shape[0]
    stats = []

    for vec_idx in range(num_vectors):
        orig_vec = original[vec_idx, :]
        recon_vec = reconstructed[vec_idx, :]

        # Original statistics
        mean_orig = np.mean(orig_vec)
        var_orig = np.var(orig_vec)
        std_orig = np.std(orig_vec)

        # Reconstructed statistics
        mean_recon = np.mean(recon_vec)
        var_recon = np.var(recon_vec)
        std_recon = np.std(recon_vec)

        # Calculate percentage errors
        mean_err = abs(mean_orig - mean_recon) / abs(mean_orig) * 100 if abs(mean_orig) > 1e-9 else 0.0
        var_err = abs(var_orig - var_recon) / abs(var_orig) * 100 if abs(var_orig) > 1e-9 else 0.0
        std_err = abs(std_orig - std_recon) / abs(std_orig) * 100 if abs(std_orig) > 1e-9 else 0.0

        stats.append({
            'vec_id': vec_idx,
            'mean_orig': mean_orig,
            'var_orig': var_orig,
            'std_orig': std_orig,
            'mean_recon': mean_recon,
            'var_recon': var_recon,
            'std_recon': std_recon,
            'mean_err_pct': mean_err,
            'var_err_pct': var_err,
            'std_err_pct': std_err
        })

    return stats


# ==========================================
# Data Collection
# ==========================================

collected_data = []

def collect_layernorm_data(module, input, output):
    global collected_data
    input_tensor = input[0].detach().cpu()

    batch_size, seq_length, hidden_dim = input_tensor.shape

    if len(collected_data) == 0:
        print(f"✓ Found LayerNorm with dimensions: {input_tensor.shape}")

    for seq_pos in range(seq_length):
        input_vector = input_tensor[0, seq_pos, :].numpy()
        collected_data.append({'input_vector': input_vector})


# ==========================================
# Main Comparison
# ==========================================

def main():
    global collected_data

    print("="*100)
    print("PER-VECTOR QUANTIZATION COMPARISON - SOLE + Alpha Factors")
    print("="*100)

    # Load model and collect data
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    # Attach hook to first LayerNorm
    hook_attached = False
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.register_forward_hook(collect_layernorm_data)
            hook_attached = True
            break

    if not hook_attached:
        print("ERROR: No LayerNorm found!")
        return

    # Run inference
    print("Loading test image...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")

    print("Running inference to collect data...")
    with torch.no_grad():
        outputs = model(**inputs)

    if len(collected_data) == 0:
        print("ERROR: No data collected!")
        return

    # Prepare data matrix
    vectors_matrix = np.array([data['input_vector'] for data in collected_data])
    print(f"✓ Collected {vectors_matrix.shape[0]} vectors × {vectors_matrix.shape[1]} channels")
    print(f"✓ Original data range: [{np.min(vectors_matrix):.6f}, {np.max(vectors_matrix):.6f}]")

    # ==========================================
    # Test All Combinations
    # ==========================================

    methods = [
        ('asymmetric', 'Asymmetric'),
        ('symmetric_fixed', 'Symmetric_Fixed_ZP128'),
        ('symmetric_corrected', 'Symmetric_Corrected')
    ]

    clipping_options = [
        (False, 'NoClip'),
        (True, f'Clip±{CLIP_THRESHOLD}')
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*100)
    print("TESTING ALL 6 COMBINATIONS")
    print("="*100)

    test_num = 1
    all_results = []

    for zp_method, zp_name in methods:
        for use_clip, clip_name in clipping_options:
            method_full_name = f"{zp_name}_{clip_name}"

            print(f"\n[Test {test_num}/6] {zp_name} + {clip_name}")
            print("-" * 100)

            # Quantize
            quantized_vectors, params = quantize_with_sole(
                vectors_matrix,
                zp_method=zp_method,
                use_clipping=use_clip
            )

            print(f"  Global S:  {params['global_s']:.8f}")
            print(f"  Global ZP: {params['global_zp']}")

            # Dequantize
            reconstructed = dequantize_with_sole(quantized_vectors, params)

            # Calculate per-vector statistics
            per_vector_stats = calculate_per_vector_statistics(vectors_matrix, reconstructed)

            # Save to file
            output_file = os.path.join(OUTPUT_DIR, f"per_vector_stats_{method_full_name}.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*150 + "\n")
                f.write(f"PER-VECTOR STATISTICS: {zp_name} + {clip_name}\n")
                f.write("="*150 + "\n")
                f.write(f"Method: {zp_method}\n")
                f.write(f"Clipping: {use_clip} (threshold: ±{CLIP_THRESHOLD if use_clip else 'N/A'})\n")
                f.write(f"Global S: {params['global_s']:.10f}\n")
                f.write(f"Global ZP: {params['global_zp']}\n")
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

            print(f"  ✓ Saved to: {output_file}")
            print(f"  Mean Error:     Avg={np.mean(mean_errs):.4f}%  Max={np.max(mean_errs):.4f}%")
            print(f"  Variance Error: Avg={np.mean(var_errs):.4f}%  Max={np.max(var_errs):.4f}%")
            print(f"  Std Error:      Avg={np.mean(std_errs):.4f}%  Max={np.max(std_errs):.4f}%")

            all_results.append({
                'method': method_full_name,
                'mean_err_avg': np.mean(mean_errs),
                'var_err_avg': np.mean(var_errs),
                'std_err_avg': np.mean(std_errs)
            })

            test_num += 1

    # ==========================================
    # Summary Comparison
    # ==========================================

    print("\n" + "="*100)
    print("SUMMARY COMPARISON - Average Errors Across All Vectors")
    print("="*100)
    print(f"\n{'Method':<40} | {'Avg Mean Err%':<15} | {'Avg Var Err%':<15} | {'Avg Std Err%':<15}")
    print("-" * 100)

    for result in all_results:
        print(f"{result['method']:<40} | {result['mean_err_avg']:>14.6f}% | "
              f"{result['var_err_avg']:>14.6f}% | {result['std_err_avg']:>14.6f}%")

    # Find best methods
    best_mean = min(all_results, key=lambda x: x['mean_err_avg'])
    best_var = min(all_results, key=lambda x: x['var_err_avg'])
    best_std = min(all_results, key=lambda x: x['std_err_avg'])

    print("\n" + "="*100)
    print("🏆 BEST METHODS")
    print("="*100)
    print(f"Best Mean Error:     {best_mean['method']} ({best_mean['mean_err_avg']:.6f}%)")
    print(f"Best Variance Error: {best_var['method']} ({best_var['var_err_avg']:.6f}%)")
    print(f"Best Std Error:      {best_std['method']} ({best_std['std_err_avg']:.6f}%)")

    # ==========================================
    # Save Unified Summary File
    # ==========================================

    summary_file = os.path.join(OUTPUT_DIR, "quantization_methods_summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("QUANTIZATION METHODS COMPARISON - UNIFIED SUMMARY\n")
        f.write("="*120 + "\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Total Vectors: {vectors_matrix.shape[0]}\n")
        f.write(f"Channels per Vector: {vectors_matrix.shape[1]}\n")
        f.write(f"Clip Threshold: ±{CLIP_THRESHOLD}\n")
        f.write("="*120 + "\n\n")

        f.write("AVERAGE ERRORS ACROSS ALL VECTORS (Lower is Better)\n")
        f.write("="*120 + "\n\n")

        # Table header
        f.write(f"{'Method':<42} | {'Avg Mean Err %':<16} | {'Avg Variance Err %':<20} | {'Avg Std Err %':<16}\n")
        f.write("-" * 120 + "\n")

        # Data rows
        for result in all_results:
            f.write(f"{result['method']:<42} | {result['mean_err_avg']:>15.8f}% | "
                   f"{result['var_err_avg']:>19.8f}% | {result['std_err_avg']:>15.8f}%\n")

        f.write("\n" + "="*120 + "\n")
        f.write("BEST METHODS BY METRIC\n")
        f.write("="*120 + "\n\n")

        f.write(f"🏆 Best Mean Error:\n")
        f.write(f"   Method: {best_mean['method']}\n")
        f.write(f"   Average Mean Error: {best_mean['mean_err_avg']:.8f}%\n\n")

        f.write(f"🏆 Best Variance Error:\n")
        f.write(f"   Method: {best_var['method']}\n")
        f.write(f"   Average Variance Error: {best_var['var_err_avg']:.8f}%\n\n")

        f.write(f"🏆 Best Std Error:\n")
        f.write(f"   Method: {best_std['method']}\n")
        f.write(f"   Average Std Error: {best_std['std_err_avg']:.8f}%\n\n")

        f.write("="*120 + "\n")
        f.write("RANKING BY MEAN ERROR (Most Important)\n")
        f.write("="*120 + "\n\n")

        sorted_by_mean = sorted(all_results, key=lambda x: x['mean_err_avg'])
        for rank, result in enumerate(sorted_by_mean, 1):
            f.write(f"{rank}. {result['method']:<40} - {result['mean_err_avg']:.8f}%\n")

        f.write("\n" + "="*120 + "\n")
        f.write("RANKING BY VARIANCE ERROR\n")
        f.write("="*120 + "\n\n")

        sorted_by_var = sorted(all_results, key=lambda x: x['var_err_avg'])
        for rank, result in enumerate(sorted_by_var, 1):
            f.write(f"{rank}. {result['method']:<40} - {result['var_err_avg']:.8f}%\n")

        f.write("\n" + "="*120 + "\n")
        f.write("RANKING BY STD ERROR\n")
        f.write("="*120 + "\n\n")

        sorted_by_std = sorted(all_results, key=lambda x: x['std_err_avg'])
        for rank, result in enumerate(sorted_by_std, 1):
            f.write(f"{rank}. {result['method']:<40} - {result['std_err_avg']:.8f}%\n")

        f.write("\n" + "="*120 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*120 + "\n\n")

        # If same method is best for multiple metrics
        if best_mean['method'] == best_var['method'] == best_std['method']:
            f.write(f"✅ CLEAR WINNER: {best_mean['method']}\n")
            f.write(f"   This method has the lowest error across ALL metrics!\n")
        else:
            f.write(f"The best method depends on your priority:\n")
            f.write(f"  - For Mean accuracy:     {best_mean['method']}\n")
            f.write(f"  - For Variance accuracy: {best_var['method']}\n")
            f.write(f"  - For Std accuracy:      {best_std['method']}\n")

    print(f"\n✓ Unified summary saved to: {summary_file}")
    print(f"✓ All per-vector statistics saved to {OUTPUT_DIR}/ directory")
    print("✓ Comparison complete!")


if __name__ == "__main__":
    main()

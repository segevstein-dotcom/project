import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import requests
import numpy as np
import math
import sys

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==========================================
# Configuration
# ==========================================
MODEL_NAME = 'facebook/deit-small-patch16-224'
CLIP_THRESHOLD = 7.0

# ==========================================
# Quantization Functions
# ==========================================

def quantize_with_sole(vectors_matrix, zp_method='symmetric_fixed', use_clipping=True):
    """
    SOLE Quantization with Alpha Factors (per-channel stretching)

    Args:
        vectors_matrix: Input float matrix (num_vectors × num_channels)
        zp_method: 'asymmetric', 'symmetric_fixed', or 'symmetric_corrected'
        use_clipping: Whether to apply outlier clipping

    Returns:
        quantized_vectors: List of quantized INT8 vectors
        params: Dictionary with quantization parameters (S, ZP, alphas)
    """
    num_vectors, num_channels = vectors_matrix.shape

    # Make a copy to avoid modifying original
    data = vectors_matrix.copy()

    # ========================================
    # Step 1: Outlier Clipping (Optional)
    # ========================================
    if use_clipping:
        data = np.clip(data, -CLIP_THRESHOLD, CLIP_THRESHOLD)

    # ========================================
    # Step 2: Calculate Global Scale (S) and Zero Point (ZP)
    # ========================================
    global_min = float(np.min(data))
    global_max = float(np.max(data))
    max_abs_value = max(abs(global_min), abs(global_max))

    if zp_method == 'asymmetric':
        # Asymmetric: use full range [min, max]
        global_s = (global_max - global_min) / 255.0
        global_zp = round(-global_min / global_s) if global_s > 0 else 128

    elif zp_method == 'symmetric_fixed':
        # Symmetric with fixed ZP=128 (SOLE original)
        global_s = max_abs_value / 127.0
        global_zp = 128

    elif zp_method == 'symmetric_corrected':
        # Symmetric with corrected ZP (dynamic)
        global_s = max_abs_value / 127.0
        normalized_min = global_min / global_s
        normalized_max = global_max / global_s
        normalized_center = (normalized_min + normalized_max) / 2
        global_zp = round(128 - normalized_center)

    else:
        raise ValueError(f"Unknown zp_method: {zp_method}")

    # ========================================
    # Step 3: Calculate Per-Channel Alpha Factors
    # ========================================
    alpha_factors = []
    processed_matrix = np.zeros_like(data)

    for channel_idx in range(num_channels):
        # Get raw values for this channel
        raw_values = data[:, channel_idx]

        # Normalize with S
        normalized_values = raw_values / global_s

        # Find max absolute value after normalization
        norm_min = float(np.min(normalized_values))
        norm_max = float(np.max(normalized_values))
        norm_max_abs = max(abs(norm_min), abs(norm_max))

        # Calculate Optimal Alpha (NEGATIVE for stretching)
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

        # Stretch and Add ZP
        # X_stretched = X_norm / 2^α
        stretched_values = normalized_values / (2 ** optimal_alpha)
        quantized_values = stretched_values + global_zp

        processed_matrix[:, channel_idx] = quantized_values

    # ========================================
    # Step 4: Quantize to INT8 [0-255]
    # ========================================
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

    Sequential Dequantization (inverse of quantization):
    1. X_stretched = X_int - ZP
    2. X_norm = X_stretched × 2^α
    3. X_real = X_norm × S
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


def calculate_errors(original, reconstructed):
    """
    Calculate error metrics between original and reconstructed data
    """
    # Flatten arrays for overall statistics
    orig_flat = original.flatten()
    recon_flat = reconstructed.flatten()

    # Calculate errors
    diff = orig_flat - recon_flat
    abs_diff = np.abs(diff)
    squared_diff = diff ** 2

    # Metrics
    mae = np.mean(abs_diff)  # Mean Absolute Error
    mse = np.mean(squared_diff)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    max_error = np.max(abs_diff)  # Maximum Error

    # Percentage error (avoid division by zero)
    nonzero_mask = orig_flat != 0
    if np.any(nonzero_mask):
        percent_error = np.mean(np.abs(diff[nonzero_mask] / orig_flat[nonzero_mask])) * 100
    else:
        percent_error = 0.0

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Max_Error': max_error,
        'Percent_Error': percent_error
    }


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

    print("="*80)
    print("QUANTIZATION METHODS COMPARISON - SOLE + Alpha Factors")
    print("="*80)

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

    print(f"\nOriginal data range: [{np.min(vectors_matrix):.6f}, {np.max(vectors_matrix):.6f}]")

    # ==========================================
    # Test All Combinations
    # ==========================================

    methods = [
        ('asymmetric', 'Asymmetric'),
        ('symmetric_fixed', 'Symmetric Fixed (ZP=128)'),
        ('symmetric_corrected', 'Symmetric Corrected')
    ]

    clipping_options = [
        (False, 'No Clipping'),
        (True, f'Clipping (±{CLIP_THRESHOLD})')
    ]

    results = []

    print("\n" + "="*80)
    print("RUNNING QUANTIZATION TESTS")
    print("="*80)

    test_num = 1
    for zp_method, zp_name in methods:
        for use_clip, clip_name in clipping_options:
            print(f"\n[Test {test_num}/6] {zp_name} + {clip_name}")
            print("-" * 80)

            # Quantize
            quantized_vectors, params = quantize_with_sole(
                vectors_matrix,
                zp_method=zp_method,
                use_clipping=use_clip
            )

            print(f"  Global S:  {params['global_s']:.8f}")
            print(f"  Global ZP: {params['global_zp']}")

            # Count alpha distribution
            alpha_dist = {}
            for alpha in params['alpha_factors']:
                alpha_dist[alpha] = alpha_dist.get(alpha, 0) + 1
            print(f"  Alpha distribution: {dict(sorted(alpha_dist.items()))}")

            # Dequantize
            reconstructed = dequantize_with_sole(quantized_vectors, params)

            # Calculate errors
            errors = calculate_errors(vectors_matrix, reconstructed)

            print(f"  MAE:           {errors['MAE']:.8f}")
            print(f"  MSE:           {errors['MSE']:.8f}")
            print(f"  RMSE:          {errors['RMSE']:.8f}")
            print(f"  Max Error:     {errors['Max_Error']:.8f}")
            print(f"  Percent Error: {errors['Percent_Error']:.4f}%")

            results.append({
                'method': zp_name,
                'clipping': clip_name,
                'global_s': params['global_s'],
                'global_zp': params['global_zp'],
                **errors
            })

            test_num += 1

    # ==========================================
    # Print Comparison Table
    # ==========================================

    print("\n" + "="*80)
    print("COMPARISON TABLE - ALL METHODS")
    print("="*80)
    print()
    print(f"{'Method':<30} {'Clipping':<20} {'MAE':<12} {'RMSE':<12} {'Max Err':<12} {'% Err':<10}")
    print("-" * 106)

    for result in results:
        print(f"{result['method']:<30} {result['clipping']:<20} "
              f"{result['MAE']:<12.8f} {result['RMSE']:<12.8f} "
              f"{result['Max_Error']:<12.8f} {result['Percent_Error']:<10.4f}%")

    # Find best method
    print("\n" + "="*80)
    print("BEST METHODS (by metric)")
    print("="*80)

    best_mae = min(results, key=lambda x: x['MAE'])
    best_rmse = min(results, key=lambda x: x['RMSE'])
    best_max = min(results, key=lambda x: x['Max_Error'])

    print(f"\n🏆 Best MAE:       {best_mae['method']} + {best_mae['clipping']}")
    print(f"   MAE = {best_mae['MAE']:.8f}")

    print(f"\n🏆 Best RMSE:      {best_rmse['method']} + {best_rmse['clipping']}")
    print(f"   RMSE = {best_rmse['RMSE']:.8f}")

    print(f"\n🏆 Best Max Error: {best_max['method']} + {best_max['clipping']}")
    print(f"   Max Error = {best_max['Max_Error']:.8f}")

    # Save results to file
    output_file = "data/quantization_comparison_results.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("QUANTIZATION METHODS COMPARISON - SOLE + Alpha Factors\n")
        f.write("="*80 + "\n\n")

        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Data shape: {vectors_matrix.shape[0]} vectors × {vectors_matrix.shape[1]} channels\n")
        f.write(f"Clip threshold: ±{CLIP_THRESHOLD}\n\n")

        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"[Test {i}] {result['method']} + {result['clipping']}\n")
            f.write(f"  Global S:      {result['global_s']:.10f}\n")
            f.write(f"  Global ZP:     {result['global_zp']}\n")
            f.write(f"  MAE:           {result['MAE']:.10f}\n")
            f.write(f"  MSE:           {result['MSE']:.10f}\n")
            f.write(f"  RMSE:          {result['RMSE']:.10f}\n")
            f.write(f"  Max Error:     {result['Max_Error']:.10f}\n")
            f.write(f"  Percent Error: {result['Percent_Error']:.6f}%\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("COMPARISON TABLE\n")
        f.write("="*80 + "\n\n")

        f.write(f"{'Method':<30} {'Clipping':<20} {'MAE':<14} {'RMSE':<14} {'Max Err':<14} {'% Err':<12}\n")
        f.write("-" * 110 + "\n")

        for result in results:
            f.write(f"{result['method']:<30} {result['clipping']:<20} "
                   f"{result['MAE']:<14.10f} {result['RMSE']:<14.10f} "
                   f"{result['Max_Error']:<14.10f} {result['Percent_Error']:<12.6f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("BEST METHODS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Best MAE:       {best_mae['method']} + {best_mae['clipping']}\n")
        f.write(f"  MAE = {best_mae['MAE']:.10f}\n\n")

        f.write(f"Best RMSE:      {best_rmse['method']} + {best_rmse['clipping']}\n")
        f.write(f"  RMSE = {best_rmse['RMSE']:.10f}\n\n")

        f.write(f"Best Max Error: {best_max['method']} + {best_max['clipping']}\n")
        f.write(f"  Max Error = {best_max['Max_Error']:.10f}\n")

    print(f"\n✓ Results saved to: {output_file}")
    print("\nComparison complete!")


if __name__ == "__main__":
    main()

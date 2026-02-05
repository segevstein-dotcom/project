import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import requests
import numpy as np
import os
import math
import sys

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==========================================
# Configuration & Constants
# ==========================================
MODEL_NAME = 'facebook/deit-small-patch16-224'
OUTPUT_DIR = "data/quantized"  # Relative to current directory (implementation/)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Output directory: {os.path.abspath(OUTPUT_DIR)}")

# ==========================================
# Helper Functions
# ==========================================

def calculate_and_quantize_with_ptf(vectors_matrix):
    """
    SEQUENTIAL QUANTIZATION APPROACH (following SOLE paper convention):
    1. S normalizes data to [-127, +127]
    2. Alpha stretches underutilized channels (alpha ≤ 0, negative for stretching)
    3. ZP shifts to [0, 255]

    Sequential steps (SOLE paper: X_int = X_real / (S × 2^α)):
    - X_norm = X_real / S
    - X_stretched = X_norm / 2^α  (when α < 0, division becomes multiplication → stretching)
    - X_int = X_stretched + ZP

    NOTE: Outlier clipping is DISABLED based on empirical results showing better accuracy without it.
          Symmetric Fixed (ZP=128) + No Clipping achieved:
          - Mean Error: 6.36% (vs 7.33% with clipping)
          - Variance Error: 0.10% (vs 0.31% with clipping)
          - Std Error: 0.05% (vs 0.17% with clipping)
    """
    print(f"Processing matrix of shape: {vectors_matrix.shape}")

    num_vectors, num_channels = vectors_matrix.shape

    # ========================================
    # NO OUTLIER CLIPPING - Better accuracy empirically verified
    # ========================================
    print(f"\n{'='*60}")
    print(f"📌 QUANTIZATION MODE: NO CLIPPING (Optimal)")
    print(f"{'='*60}")
    print(f"Using full data range without clipping")
    print(f"Data range: [{np.min(vectors_matrix):.6f}, {np.max(vectors_matrix):.6f}]")
    print(f"{'='*60}")

    # --- Step 1: Calculate Global S to normalize to ±127 ---
    global_min = float(np.min(vectors_matrix))
    global_max = float(np.max(vectors_matrix))
    max_abs_value = max(abs(global_min), abs(global_max))

    # Symmetric quantization: ZP=128, S from max absolute value
    SIGNED_8BIT_MAX = 127
    global_s = max_abs_value / SIGNED_8BIT_MAX
    global_zp = 128

    print(f"\n{'='*60}")
    print(f"📊 GLOBAL QUANTIZATION PARAMETERS")
    print(f"{'='*60}")
    print(f"   S (Scale):     {global_s:.8f}")
    print(f"   ZP (Zero Point): {global_zp}")
    print(f"{'='*60}")

    alpha_factors = []
    channel_stats = []

    processed_matrix = np.zeros_like(vectors_matrix)

    for channel_idx in range(num_channels):

        # 1. Get raw values for this channel
        raw_values = vectors_matrix[:, channel_idx]

        # 2. SEQUENTIAL STEP 1: Normalize with S
        normalized_values = raw_values / global_s

        # 3. Find max absolute value after normalization
        norm_min = float(np.min(normalized_values))
        norm_max = float(np.max(normalized_values))
        norm_max_abs = max(abs(norm_min), abs(norm_max))

        # 4. Calculate Optimal Alpha (ALWAYS ≤ 0, NEGATIVE for stretching)
        # After normalization, values are in roughly [-127, +127]
        # We want to stretch underutilized channels
        # SOLE paper: X_stretched = X_norm / 2^α (when α < 0, this multiplies → stretching)
        # Find most negative alpha where: norm_max_abs / 2^α ≤ 127

        if norm_max_abs < 1e-9:
            optimal_alpha = 0
        else:
            # Maximum safe stretch: 127 / norm_max_abs
            max_stretch = 127.0 / norm_max_abs

            # Find largest integer abs_alpha where 2^abs_alpha ≤ max_stretch
            if max_stretch >= 1.0:
                abs_alpha = int(math.floor(math.log2(max_stretch)))
            else:
                abs_alpha = 0  # Can't stretch, already utilizing full range

            # Make alpha NEGATIVE for stretching (SOLE paper convention)
            optimal_alpha = -abs_alpha

        alpha_factors.append(optimal_alpha)

        # 5. SEQUENTIAL STEP 2 & 3: Stretch and Add ZP
        # SOLE paper formula: X_stretched = X_norm / 2^α
        # When α is negative (e.g., -2), dividing by 2^(-2) = dividing by 0.25 = multiplying by 4 (stretching!)
        # X_int = X_stretched + ZP
        stretched_values = normalized_values / (2 ** optimal_alpha)
        quantized_values = stretched_values + global_zp

        channel_stats.append({
            'channel': channel_idx,
            'norm_max_abs': norm_max_abs,
            'alpha': optimal_alpha,
            'stretch_factor': 2 ** optimal_alpha,
            'global_zp': global_zp
        })

        processed_matrix[:, channel_idx] = quantized_values

    # Quantize to INT8 [0-255]
    quantized_vectors = []
    quantization_stats = []
    
    for vector_idx in range(num_vectors):
        row_values = processed_matrix[vector_idx, :]
        
        # Round and Clip to uint8 [0-255]
        quantized_vector = np.round(row_values).astype(np.int32)
        quantized_vector = np.clip(quantized_vector, 0, 255)
        
        quantized_vectors.append(quantized_vector)
        
        max_used = np.max(quantized_vector)
        quantization_stats.append({
            'vector_idx': vector_idx,
            'max_quantized_value': int(max_used),
            'bit_utilization': max_used / 255.0
        })
    
    # Analyze Alpha distribution
    alpha_distribution = {}
    for alpha in alpha_factors:
        alpha_distribution[alpha] = alpha_distribution.get(alpha, 0) + 1
    
    print(f"\nAlpha factor distribution:")
    for alpha_val in sorted(alpha_distribution.keys()):
        count = alpha_distribution[alpha_val]
        percentage = (count / len(alpha_factors)) * 100
        print(f"  Alpha {alpha_val}: {count} channels ({percentage:.1f}%)")
    
    return quantized_vectors, alpha_factors, global_s, global_zp, quantization_stats

def collect_layernorm_data(module, input, output):
    global collected_data
    input_tensor = input[0].detach().cpu()
    
    gamma = None; beta = None
    if hasattr(module, 'weight') and module.weight is not None:
        gamma = module.weight.detach().cpu().numpy()
    if hasattr(module, 'bias') and module.bias is not None:
        beta = module.bias.detach().cpu().numpy()
    
    batch_size, seq_length, hidden_dim = input_tensor.shape
    
    if len(collected_data) == 0:
        print(f"Found LayerNorm with dimensions: {input_tensor.shape}")
    
    for seq_pos in range(seq_length):
        input_vector = input_tensor[0, seq_pos, :].numpy()
        collected_data.append({
            'input_vector': input_vector, 'gamma': gamma, 'beta': beta,
            'hidden_dim': hidden_dim, 'seq_position': seq_pos
        })

def save_ptf_data(raw_matrix, quantized_vectors, alpha_factors, global_s, global_zp, gamma, beta, quantization_stats):

    # 1. Export Raw Floats
    raw_dir = os.path.join(OUTPUT_DIR, "raw_input_vectors")
    os.makedirs(raw_dir, exist_ok=True)
    print(f"Saving RAW float vectors to {raw_dir}/...")
    for i in range(raw_matrix.shape[0]):
        vec = raw_matrix[i, :]
        with open(os.path.join(raw_dir, f"raw_vec_{i:03d}.txt"), 'w') as f:
            f.write(f"{len(vec)}\n")
            for val in vec: f.write(f"{val:.8f}\n")

    # 2. Export Alpha Factors
    with open(os.path.join(OUTPUT_DIR, "alpha_factors.txt"), 'w') as f:
        f.write(f"# Channel-wise alpha factors\n{len(alpha_factors)}\n")
        for alpha in alpha_factors: f.write(f"{alpha}\n")
    
    
    # 4. Export Global Parameters (UPDATED with S and ZP)
    with open(os.path.join(OUTPUT_DIR, "global_params.txt"), 'w') as f:
        f.write(f"# hidden_dim, num_vectors, global_s, global_zp\n")
        f.write(f"{len(alpha_factors)}\n")
        f.write(f"{len(quantized_vectors)}\n")
        f.write(f"{global_s:.10f}\n") # High precision for Scale
        f.write(f"{global_zp}\n")
    
    # 5. Export Learned Weights (Float - Original)
    with open(os.path.join(OUTPUT_DIR, "layernorm_weights.txt"), 'w') as f:
        f.write(f"{len(alpha_factors)}\n")
        if gamma is not None:
            for val in gamma: f.write(f"{val:.8f}\n")
        else:
             for _ in range(len(alpha_factors)): f.write("1.0\n")
        if beta is not None:
            for val in beta: f.write(f"{val:.8f}\n")
        else:
             for _ in range(len(alpha_factors)): f.write("0.0\n")

    # 5b. Quantize and Export Gamma & Beta (8-bit per-channel)
    if gamma is not None and beta is not None:
        # Quantize gamma (affine weight)
        gamma_min, gamma_max = float(np.min(gamma)), float(np.max(gamma))
        gamma_scale = (gamma_max - gamma_min) / 255.0
        gamma_zp = int(round(-gamma_min / gamma_scale))
        gamma_quantized = np.clip(np.round(gamma / gamma_scale) + gamma_zp, 0, 255).astype(np.uint8)

        # Quantize beta (affine bias)
        beta_min, beta_max = float(np.min(beta)), float(np.max(beta))
        beta_scale = (beta_max - beta_min) / 255.0
        beta_zp = int(round(-beta_min / beta_scale))
        beta_quantized = np.clip(np.round(beta / beta_scale) + beta_zp, 0, 255).astype(np.uint8)

        # Export quantized gamma
        with open(os.path.join(OUTPUT_DIR, "gamma_quantized.txt"), 'w') as f:
            f.write(f"# Quantized Gamma (8-bit per-channel)\n")
            f.write(f"# Scale: {gamma_scale:.10f}, ZP: {gamma_zp}\n")
            f.write(f"{len(gamma_quantized)}\n")
            f.write(f"{gamma_scale:.10f}\n")
            f.write(f"{gamma_zp}\n")
            for val in gamma_quantized: f.write(f"{int(val)}\n")

        # Export quantized beta
        with open(os.path.join(OUTPUT_DIR, "beta_quantized.txt"), 'w') as f:
            f.write(f"# Quantized Beta (8-bit per-channel)\n")
            f.write(f"# Scale: {beta_scale:.10f}, ZP: {beta_zp}\n")
            f.write(f"{len(beta_quantized)}\n")
            f.write(f"{beta_scale:.10f}\n")
            f.write(f"{beta_zp}\n")
            for val in beta_quantized: f.write(f"{int(val)}\n")

        print(f"✓ Gamma quantized: scale={gamma_scale:.6f}, ZP={gamma_zp}, range=[{gamma_min:.4f}, {gamma_max:.4f}]")
        print(f"✓ Beta quantized:  scale={beta_scale:.6f}, ZP={beta_zp}, range=[{beta_min:.4f}, {beta_max:.4f}]")
    
    # 6. Export Quantized Vectors
    for i, quantized_vector in enumerate(quantized_vectors):
        with open(os.path.join(OUTPUT_DIR, f"vector_{i:03d}.txt"), 'w') as f:
            f.write(f"# SOLE Quantized Vector {i}\n{len(quantized_vector)}\n")
            for val in quantized_vector: f.write(f"{int(val)}\n")

    # 7. Export Statistics
    with open(os.path.join(OUTPUT_DIR, "quantization_utilization.txt"), 'w') as f:
        for stat in quantization_stats:
            f.write(f"{stat['vector_idx']}, {stat['max_quantized_value']}, {stat['bit_utilization']*100:.1f}%\n")

def generate_golden_reference(original_vector, quantized_vector, global_zp, global_s, alpha_factors, gamma, beta, vector_idx=0):
    """
    Generates golden reference using ORIGINAL input statistics.
    The mean/variance should be from the ORIGINAL input (before quantization),
    not from the reconstructed (dequantized) values.

    SEQUENTIAL DEQUANTIZATION (inverse of sequential quantization, SOLE paper):
    - X_stretched = X_int - ZP
    - X_norm = X_stretched × 2^α  (when α is negative, this divides → unstretch)
    - X_real = X_norm × S
    """
    print(f"\nGenerating Golden Reference for Vector #{vector_idx}...")

    # 1. Calculate Expected Stats from ORIGINAL input (before quantization)
    golden_mean = np.mean(original_vector)
    golden_var = np.var(original_vector)
    golden_std = np.sqrt(golden_var)

    # 2. Reconstruct values using SEQUENTIAL dequantization
    reconstructed_values = []
    for i in range(len(quantized_vector)):
        q_val = int(quantized_vector[i])
        zp = int(global_zp)
        alpha = int(alpha_factors[i])
        s = float(global_s)

        # Sequential Dequantization (SOLE paper, inverse of quantization):
        # Step 1: Remove ZP
        x_stretched = q_val - zp
        # Step 2: Unstretch (multiply by 2^α, when α negative this divides)
        x_norm = x_stretched * (2 ** alpha)
        # Step 3: Denormalize (multiply by S)
        x_real = x_norm * s

        reconstructed_values.append(x_real)

    reconstructed_values = np.array(reconstructed_values)

    # Recalculate mean/var from reconstructed for comparison
    reconstructed_mean = np.mean(reconstructed_values)
    reconstructed_var = np.var(reconstructed_values)

    # 3. Determine which affine params to use: prefer dequantized quantized-gamma/beta
    gamma_deq = None
    beta_deq = None

    gamma_q_path = os.path.join(OUTPUT_DIR, "gamma_quantized.txt")
    beta_q_path = os.path.join(OUTPUT_DIR, "beta_quantized.txt")

    def _read_and_dequantize(path):
        try:
            with open(path, 'r') as fh:
                lines = [l.strip() for l in fh.readlines() if l.strip() != '']
            # remove comment lines
            data_lines = [l for l in lines if not l.startswith('#')]
            if len(data_lines) < 4:
                return None
            count = int(data_lines[0])
            scale = float(data_lines[1])
            zp = int(data_lines[2])
            vals = [int(x) for x in data_lines[3:3+count]]
            deq = np.array([(v - zp) * scale for v in vals], dtype=np.float64)
            return deq
        except Exception:
            return None

    gamma_deq = _read_and_dequantize(gamma_q_path)
    beta_deq = _read_and_dequantize(beta_q_path)

    if gamma_deq is None or beta_deq is None:
        # Fallback to provided float params if quantized files missing
        if gamma is None:
            gamma_deq = np.ones(len(alpha_factors), dtype=np.float64)
        else:
            gamma_deq = np.array(gamma, dtype=np.float64)
        if beta is None:
            beta_deq = np.zeros(len(alpha_factors), dtype=np.float64)
        else:
            beta_deq = np.array(beta, dtype=np.float64)
        print("⚠️  Using original float gamma/beta for golden (quantized files not found)")
    else:
        print("✓ Using dequantized gamma/beta from quantized export for golden reference")

    # 4. Calculate Expected Output (Stage 2) using ORIGINAL statistics
    std_inv = 1.0 / np.sqrt(golden_var + 1e-6)
    golden_output = (reconstructed_values - golden_mean) * std_inv * gamma_deq + beta_deq

    # 4. Print to Console
    print("="*40)
    print("🌟 GOLDEN REFERENCE 🌟")
    print("="*40)
    print(f"Global Scale (S):     {global_s:.8f}")
    print(f"Global ZP:            {global_zp}")
    print(f"ORIGINAL Mean:        {golden_mean:.6f}")
    print(f"ORIGINAL Variance:    {golden_var:.6f}")
    print(f"ORIGINAL Std:         {golden_std:.6f}")
    print(f"Reconstructed Mean:   {reconstructed_mean:.6f} (after quant+dequant)")
    print(f"Reconstructed Var:    {reconstructed_var:.6f} (after quant+dequant)")
    print(f"Mean Error:           {abs(golden_mean - reconstructed_mean):.6f}")
    print(f"Var Error:            {abs(golden_var - reconstructed_var):.6f}")
    print("-" * 20)
    print("First 5 Expected Outputs:")
    for i in range(5):
        print(f"  [{i}] {golden_output[i]:.4f}")
    print("="*40)
    
    # 5. Save to file
    with open(os.path.join(OUTPUT_DIR, "golden_ref_vec000.txt"), "w") as f:
        f.write(f"# Golden Reference for Vector 0\n")
        f.write(f"Global S: {global_s:.10f}\n")
        f.write(f"Global ZP: {global_zp}\n")
        f.write(f"ORIGINAL Mean: {golden_mean:.6f}\n")
        f.write(f"ORIGINAL Variance: {golden_var:.6f}\n")
        f.write(f"Reconstructed Mean: {reconstructed_mean:.6f}\n")
        f.write(f"Reconstructed Variance: {reconstructed_var:.6f}\n")
        f.write("Output:\n")
        for val in golden_output:
            f.write(f"{val:.6f}\n")
            
def main():
    global collected_data
    collected_data = []
    
    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    
    hook_attached = False
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            module.register_forward_hook(collect_layernorm_data)
            hook_attached = True; break
    if not hook_attached: return

    print("\nLoading test image...")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad(): outputs = model(**inputs)
    if len(collected_data) == 0: return

    vectors_matrix = np.array([data['input_vector'] for data in collected_data])
    
    print(f"\n" + "="*60)
    print("STARTING SOLE QUANTIZATION (Global S/ZP, Per-Channel Alpha)")
    print("="*60)
    
    # Call updated function
    quantized_vectors, alpha_factors, global_s, global_zp, quantization_stats = calculate_and_quantize_with_ptf(vectors_matrix)
    
    gamma = collected_data[0]['gamma']
    beta = collected_data[0]['beta']
    
    print(f"\nSaving generated files to {OUTPUT_DIR}...")
    save_ptf_data(vectors_matrix, quantized_vectors, alpha_factors, global_s, global_zp, gamma, beta, quantization_stats)

    # Pass ORIGINAL vector and quantized vector to golden reference generator
    generate_golden_reference(vectors_matrix[0], quantized_vectors[0], global_zp, global_s, alpha_factors, gamma, beta, vector_idx=0)

    print("Processing complete.")

if __name__ == "__main__":
    main()
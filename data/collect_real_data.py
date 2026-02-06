
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
    """
    print(f"Processing matrix of shape: {vectors_matrix.shape}")

    num_vectors, num_channels = vectors_matrix.shape

    # ========================================
    # OUTLIER CLIPPING (Optional but Recommended)
    # ========================================
    # Clip extreme outliers to improve quantization precision for majority of values
    CLIP_THRESHOLD = 7.0  # Conservative: clips only top 1 outlier (59.5% S improvement!)

    # Before clipping statistics
    raw_min = float(np.min(vectors_matrix))
    raw_max = float(np.max(vectors_matrix))

    # Apply clipping
    vectors_matrix = np.clip(vectors_matrix, -CLIP_THRESHOLD, CLIP_THRESHOLD)

    # After clipping statistics
    num_clipped = np.sum((vectors_matrix == CLIP_THRESHOLD) | (vectors_matrix == -CLIP_THRESHOLD))

    print(f"\n{'='*60}")
    print(f"📌 OUTLIER CLIPPING")
    print(f"{'='*60}")
    print(f"Threshold: ±{CLIP_THRESHOLD}")
    print(f"Before: min={raw_min:.6f}, max={raw_max:.6f}")
    print(f"After:  min={np.min(vectors_matrix):.6f}, max={np.max(vectors_matrix):.6f}")
    print(f"Values clipped: {num_clipped} / {vectors_matrix.size} ({num_clipped/vectors_matrix.size*100:.4f}%)")
    print(f"S improvement: {(max(abs(raw_min), abs(raw_max)) / max(abs(np.min(vectors_matrix)), abs(np.max(vectors_matrix))) - 1) * 100:.2f}%")
    print(f"{'='*60}")

    # --- Step 1: Calculate Global S to normalize to ±127 ---
    global_min = float(np.min(vectors_matrix))
    global_max = float(np.max(vectors_matrix))

    # ========================================
    # OPTION 1: ASYMMETRIC QUANTIZATION
    # ========================================
    # Uses full range [min, max] for maximum precision
    asymmetric_scale = (global_max - global_min) / 255.0
    asymmetric_zp = round(-global_min / asymmetric_scale) if asymmetric_scale > 0 else 128

    # ========================================
    # OPTION 2: SYMMETRIC QUANTIZATION (SOLE paper style)
    # ========================================
    # Symmetric quantization: use max absolute value
    max_abs_value = max(abs(global_min), abs(global_max))

    # S normalizes to ±127
    SIGNED_8BIT_MAX = 127
    global_s = max_abs_value / SIGNED_8BIT_MAX

    # Calculate ZP dynamically based on actual data distribution
    normalized_min = global_min / global_s
    normalized_max = global_max / global_s

    # For symmetric: center the normalized range in [0, 255]
    normalized_center = (normalized_min + normalized_max) / 2
    symmetric_zp_corrected = round(128 - normalized_center)

    # OLD approach (assumes data centered at 0)
    symmetric_zp_old = 128

    # ========================================
    # COMPARISON OF METHODS
    # ========================================
    print(f"\n{'='*60}")
    print(f"⚖️  COMPARING QUANTIZATION METHODS")
    print(f"{'='*60}")
    print(f"\n📊 Option 1: ASYMMETRIC QUANTIZATION")
    print(f"   Scale: {asymmetric_scale:.8f}")
    print(f"   ZP:    {asymmetric_zp}")
    print(f"   Maps [{global_min:.6f}, {global_max:.6f}] → [0, 255]")
    print(f"   Quantized range check:")
    print(f"      min_quantized = ({global_min:.4f} / {asymmetric_scale:.6f}) + {asymmetric_zp} = {global_min/asymmetric_scale + asymmetric_zp:.2f}")
    print(f"      max_quantized = ({global_max:.4f} / {asymmetric_scale:.6f}) + {asymmetric_zp} = {global_max/asymmetric_scale + asymmetric_zp:.2f}")

    print(f"\n📊 Option 2: SYMMETRIC (SOLE style) with CORRECTED ZP")
    print(f"   Scale (S): {global_s:.8f}")
    print(f"   ZP:        {symmetric_zp_corrected}")
    print(f"   Normalized range: [{normalized_min:.2f}, {normalized_max:.2f}]")
    print(f"   After ZP shift: [{normalized_min + symmetric_zp_corrected:.2f}, {normalized_max + symmetric_zp_corrected:.2f}]")

    print(f"\n📊 OLD Method: SYMMETRIC with FIXED ZP=128")
    print(f"   Scale (S): {global_s:.8f}")
    print(f"   ZP:        {symmetric_zp_old}")
    print(f"   After ZP shift: [{normalized_min + 128:.2f}, {normalized_max + 128:.2f}]")
    if abs(normalized_center) > 1e-3:
        print(f"   ⚠️  WARNING: Data not centered! Center offset = {normalized_center:.4f}")
        print(f"   ⚠️  Using ZP=128 will waste {abs(normalized_center):.2f} quantization levels!")

    print(f"\n🎯 RECOMMENDATION:")
    if abs(normalized_center) < 1.0:
        print(f"   Data is relatively centered. Both symmetric options should work.")
        print(f"   Difference: {abs(symmetric_zp_corrected - 128)} quantization levels")
    else:
        print(f"   ⚠️  Data has significant bias!")
        print(f"   Asymmetric quantization will be more accurate.")
    print(f"{'='*60}")

    # DECISION: Use Q1 - SOLE Original (ZP=128) - Best results based on 8-bit comparison
    # Q1 showed lowest variance error (~0.1%) in quantization_comparison_8bit.txt
    global_zp = 128  # symmetric_zp_old - SOLE original method

    print(f"\n{'='*60}")
    print(f"📊 SEQUENTIAL QUANTIZATION - STEP 1: NORMALIZE WITH S")
    print(f"{'='*60}")
    print(f"Data range:")
    print(f"   Min value: {global_min:.6f}")
    print(f"   Max value: {global_max:.6f}")
    print(f"   Max absolute: {max_abs_value:.6f}")
    print(f"\nS calculation (normalize to ±127):")
    print(f"   S = max_abs / 127")
    print(f"   S = {max_abs_value:.6f} / 127")
    print(f"   S = {global_s:.8f}")
    print(f"\nAfter X_norm = X_real / S:")
    print(f"   Normalized range: [{global_min/global_s:.2f}, {global_max/global_s:.2f}]")
    print(f"   (Should be approximately [-127, +127])")
    print(f"{'='*60}")

    print(f"\n🌎 GLOBAL PARAMETERS:")
    print(f"   S (Scale Factor): {global_s:.8f}")
    print(f"   ZP (Zero Point):  {global_zp}")

    alpha_factors = []
    channel_stats = []

    processed_matrix = np.zeros_like(vectors_matrix)
    
    print(f"\n{'='*60}")
    print(f"📊 STEP 2: CALCULATE PER-CHANNEL ALPHA (STRETCH FACTORS)")
    print(f"{'='*60}")
    print(f"Processing {num_channels} channels...")

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

    # --- Quantization Step ---
    print(f"\nQuantizing {num_vectors} vectors...")
    
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
        # gamma_min, gamma_max = float(np.min(gamma)), float(np.max(gamma)) 
        # אנחנו לוקחים גבולות קצת יותר הדוקים כדי לשפר את הדיוק של הרוב
        gamma_min = np.percentile(gamma, 0.1)  
        gamma_max = np.percentile(gamma, 99.9) 
        gamma_scale = (gamma_max - gamma_min) / 255.0
        gamma_zp = int(round(-gamma_min / gamma_scale))
        gamma_quantized = np.clip(np.round(gamma / gamma_scale) + gamma_zp, 0, 255).astype(np.uint8)

        # Quantize beta (affine bias)
        # beta_min, beta_max = float(np.min(beta)), float(np.max(beta))
        beta_min = np.percentile(beta, 0.1) 
        beta_max = np.percentile(beta, 99.9) 
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


def generate_golden_reference(original_vector, quantized_vector, global_zp, global_s, alpha_factors, gamma, beta, vector_idx):
    """
    מפיקה Golden Reference עבור וקטור ספציפי.
    הפונקציה מבצעת דקוונטיזציה מלאה ומחשבת את פלט ה-LayerNorm 
    בהתבסס על הסטטיסטיקה של הוקטור המקורי (לפני קוונטיזציה).
    """
    
    # 1. חישוב סטטיסטיקה מקורית (Gold Standard)
    golden_mean = np.mean(original_vector)
    golden_var = np.var(original_vector)
    std_inv = 1.0 / np.sqrt(golden_var + 1e-6)
    
    # 2. שחזור ערכים (Sequential Dequantization לפי ה-SOLE Paper)
    # X_real = ((X_int - ZP) * 2^alpha) * S
    reconstructed = []
    for i in range(len(quantized_vector)):
        x_stretched = int(quantized_vector[i]) - int(global_zp)
        x_norm = x_stretched * (2.0 ** int(alpha_factors[i]))
        reconstructed.append(x_norm * float(global_s))
    
    reconstructed = np.array(reconstructed)
    
    # 3. חישוב פלט ה-LayerNorm הסופי (Stage 2)
    # Y = (X_reconstructed - Mean_original) * StdInv_original * Gamma + Beta
    golden_output = (reconstructed - golden_mean) * std_inv * gamma + beta
    
    # 4. הדפסת סיכום עבור הווקטור הראשון בלבד (לצורך בדיקת שפיות)
    if vector_idx == 0:
        print(f"\n" + "="*40)
        print(f"🌟 GOLDEN REFERENCE SUMMARY (Vector #{vector_idx})")
        print(f"="*40)
        print(f"Original Mean:    {golden_mean:.6f}")
        print(f"Original Var:     {golden_var:.6f}")
        print(f"Reconstructed Mean: {np.mean(reconstructed):.6f}")
        print(f"Reconstructed Var:  {np.var(reconstructed):.6f}")
        print("-" * 40)

    # 5. שמירה לקובץ
    file_path = os.path.join(OUTPUT_DIR, f"golden_ref_vec{vector_idx:03d}.txt")
    with open(file_path, "w", encoding='utf-8') as f:
        f.write(f"ORIGINAL Mean: {golden_mean:.6f}\n")
        f.write(f"ORIGINAL Variance: {golden_var:.6f}\n")
        f.write("Output:\n")
        for val in golden_output:
            f.write(f"{val:.6f}\n")
            
    return golden_mean, golden_var


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
    # Example image URLs from COCO dataset
    # url = "http://images.cocodataset.org/val2017/000000000632.jpg"
    url ="http://images.cocodataset.org/val2017/000000000139.jpg"
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
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

    print(f"\nGenerating Golden References for {len(quantized_vectors)} vectors...")
    for i in range(len(quantized_vectors)):
        generate_golden_reference(vectors_matrix[i], quantized_vectors[i], 
                                  global_zp, global_s, alpha_factors, 
                                  gamma, beta, vector_idx=i)

    print(f"✓ Created {len(quantized_vectors)} golden reference files in {OUTPUT_DIR}")


    print("Processing complete. All golden files generated.")
    
    
    print("Processing complete.")
    
    
if __name__ == "__main__":
    main()
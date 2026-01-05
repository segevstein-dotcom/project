# Quantization Experiments

This folder contains older quantization experiments and comparisons that were used to evaluate different quantization methods.

## Files Overview

### Method Comparison Files

#### `per_vector_stats_*.txt` (7 files)
Detailed per-vector statistics for different quantization methods:

1. **Asymmetric_Clip±7.0.txt** - Asymmetric quantization with clipping at ±7σ
2. **Asymmetric_NoClip.txt** - Asymmetric quantization without clipping
3. **Symmetric_Corrected_Clip±7.0.txt** - Symmetric with corrected ZP, clipped at ±7σ
4. **Symmetric_Corrected_NoClip.txt** - Symmetric with corrected ZP, no clipping
5. **Symmetric_Fixed_ZP128_Clip±7.0.txt** - Symmetric with fixed ZP=128, clipped at ±7σ
6. **Symmetric_Fixed_ZP128_NoClip.txt** - Symmetric with fixed ZP=128, no clipping

Each file contains:
- Per-vector mean, variance, std (original and reconstructed)
- Quantization errors
- Utilization statistics

---

### Comparison Reports

#### `quantization_comparison_results.txt`
Summary comparison of different quantization methods showing:
- Mean absolute errors
- Maximum errors
- Utilization rates
- Best performing method

#### `quantization_methods_summary.txt`
Detailed summary of each quantization method's performance:
- Mean errors across all vectors
- Max errors
- Bit utilization
- Recommendations

---

### Validation Files

#### `c_vs_golden_quantize_comparison.txt`
Comparison between C implementation and golden reference using dequantized values.
- **Note:** Uses dequantized values as golden reference (not raw inputs)
- Shows lower errors because both sides have quantization loss

#### `quantization_validation.txt`
Per-vector validation of quantization quality:
- Original values
- Quantized values
- Reconstructed values
- Errors

#### `quantize_validation_summary.txt`
Summary statistics for quantization validation:
- Overall mean errors
- Max errors
- Variance in errors

---

## Historical Context

These files were created during the quantization method selection phase when evaluating:
- **Symmetric vs Asymmetric** quantization
- **Fixed vs Corrected** zero-point
- **Clipping vs No Clipping** strategies

### Selected Method (Current Implementation)
Based on these experiments, the final implementation uses:
- **Symmetric quantization** with fixed ZP=128
- **No clipping** (to preserve full dynamic range)
- **PTF (Power-of-Two Factor)** for per-channel scaling
- **Global scale factor** for overall normalization

This is implemented in:
- `src/collect_real_data.py` - Data generation with selected method
- `src/main.c` - C implementation using the quantized data

---

## How These Differ from Current Validation

**These experiments:**
- Compare different **quantization strategies**
- Focus on **quantization error** (float → uint8 → float)
- Golden reference = dequantized values

**Current validation (`validation/`):**
- Validates the **SOLE algorithm implementation**
- Compares C code vs **original raw inputs** (absolute ground truth)
- Includes both quantization errors AND algorithm approximations (dynamic compression, square LUT, etc.)

---

## Files Not Needed for Current Work

These files are kept for **historical reference** and **reproducibility** but are not used in the current validation workflow.

For current validation, use: `validation/validate_statistics.py`

---

## Folder Structure

```
quantization_experiments/
├── README.md (this file)
├── per_vector_stats_Asymmetric_Clip±7.0.txt
├── per_vector_stats_Asymmetric_NoClip.txt
├── per_vector_stats_Symmetric_Corrected_Clip±7.0.txt
├── per_vector_stats_Symmetric_Corrected_NoClip.txt
├── per_vector_stats_Symmetric_Fixed_ZP128_Clip±7.0.txt
├── per_vector_stats_Symmetric_Fixed_ZP128_NoClip.txt
├── quantization_comparison_results.txt
├── quantization_methods_summary.txt
├── c_vs_golden_quantize_comparison.txt
├── quantization_validation.txt
└── quantize_validation_summary.txt
```

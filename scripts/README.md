# Data Generation Scripts

This folder contains Python scripts for generating quantized data used by the C implementation.

## Scripts

### `collect_real_data.py`
**Main data generation script** - Generates PTF-quantized data from DeiT model.

**Usage:**
```bash
# Run from implementation/ directory
python scripts/collect_real_data.py
```

**What it does:**
1. Loads DeiT-small model from HuggingFace
2. Extracts LayerNorm input vectors from test image
3. Performs PTF (Power-of-Two Factor) quantization:
   - Calculates global scale factor (S) and zero point (ZP)
   - Computes per-channel alpha factors for optimal bit utilization
   - Quantizes to 8-bit unsigned integers [0-255]
4. Generates all necessary data files in `data/quantized/`:
   - `raw_input_vectors/` - Original float32 values (ground truth)
   - `quantized_vectors/` - PTF-quantized uint8 values
   - `alpha_factors.txt` - Per-channel scaling factors
   - `global_params.txt` - S, ZP, dimensions
   - `layernorm_weights.txt` - Gamma and beta (float)
   - `gamma_quantized.txt`, `beta_quantized.txt` - Quantized weights
   - `golden_ref_vec000.txt` - Reference output for validation

**Quantization Method:**
- **Sequential PTF Quantization** (SOLE paper):
  1. Normalize: `X_norm = X_real / S`
  2. Stretch: `X_stretched = X_norm / 2^α` (α ≤ 0 for stretching)
  3. Shift: `X_int = X_stretched + ZP`

**Configuration:**
- Model: `facebook/deit-small-patch16-224`
- Hidden dimension: 384
- Number of vectors: 197 (sequence length)
- Quantization: 8-bit symmetric (ZP=128)
- No outlier clipping (empirically better accuracy)

**Output:**
All files are written to `data/quantized/` directory, ready for C code consumption.

---

## Adding New Scripts

When adding new data generation scripts:
1. Place them in this folder
2. Document them in this README
3. Use relative paths: `data/...` (run from `implementation/`)
4. Add clear usage instructions

---

## Notes

- All scripts should be run from the `implementation/` directory
- Scripts use relative paths assuming `implementation/` as working directory
- Generated data is consumed by C code in `src/`
- Validation scripts in `validation/` verify the generated data

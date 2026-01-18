# Inverse Square Root LUT Implementation

## Overview

Hardware-accelerated **Inverse Square Root Lookup Table (LUT)** for LayerNorm computation.
Approximates `f(x) = 1/√x` using **256 8-bit entries**, achieving **0.288% mean error**.

---

## Quick Start

### Step 1: Analyze Data Distribution
```bash
python scripts/analyze_var_hw_distribution.py
```
**Output:** `data/lut/var_hw_analysis.txt` (statistical analysis of 197 vectors)

### Step 2: Generate LUT
```bash
python scripts/generate_inv_sqrt_lut.py
```
**Output:** `data/lut/lut_inv_sqrt.txt` (256-entry LUT for C code)

### Step 3: Validate Parameters (Grid Search)
```bash
python scripts/grid_search_lut_params.py
```
**Output:**
- `data/lut/grid_search_results.txt` (detailed comparison)
- `data/lut/lut_grid_search_heatmap.png` (MSE visualization)
- `data/lut/lut_grid_search_rel_error.png` (relative error visualization)

---

## Results

### Optimal Parameters
| Parameter | Value | Binary | Notes |
|-----------|-------|--------|-------|
| **INPUT_DIVISOR** | 128 | 2^7 | Maps var_hw to index [0-255] |
| **OUTPUT_SCALE** | 16384 | 2^14 | Maps 8-bit value to float |
| **LUT Size** | 256 | - | 8-bit addressing |
| **Data Width** | 8-bit | - | uint8_t values |

### Accuracy
| Metric | Value |
|--------|-------|
| **Mean Relative Error** | **0.288%** |
| Max Relative Error | 0.795% |
| MSE | 9.26 × 10⁻¹⁰ |
| LUT Utilization | 66.6% (indices 86-169) |

### Grid Search Results (25 Combinations Tested)

**TOP 5:**
| Rank | INPUT_DIV | OUTPUT_SCALE | MSE | Error % |
|------|-----------|--------------|-----|---------|
| **1** | **128** | **16384** | **9.26e-10** | **0.288%** ⭐ |
| 2 | 128 | 8192 | 1.89e-09 | 0.424% |
| 3 | 256 | 16384 | 2.23e-09 | 0.458% |
| 4 | 256 | 8192 | 3.06e-09 | 0.535% |
| 5 | 128 | 4096 | 6.45e-09 | 0.795% |

Our choice is **mathematically optimal** among all tested configurations.

---

## Usage in C

### LUT Definition ([utils.h](src/utils.h))
```c
#define INV_SQRT_LUT_SIZE 256
#define INV_SQRT_INPUT_DIVISOR 128    // 2^7
#define INV_SQRT_OUTPUT_SCALE 16384   // 2^14
extern uint8_t INV_SQRT_LUT[INV_SQRT_LUT_SIZE];
```

### Loading LUT ([utils.c](src/utils.c))
```c
int load_inv_sqrt_lut(void);  // Call once at startup
```

### Using LUT
```c
float inv_sqrt_lut(float var_hw) {
    int index = (int)(var_hw / 128);
    if (index > 255) index = 255;
    uint8_t lut_val = INV_SQRT_LUT[index];
    return (float)lut_val / 16384.0;
}
```

**Example:**
```c
var_hw = 13500.0;
inv_std = inv_sqrt_lut(var_hw);  // ≈ 0.00861
```

---

## Technical Details

### Why These Parameters?

**INPUT_DIVISOR = 128:**
- var_hw range: [11,118, 21,746] (from real data)
- Index range: [86.9, 169.9] → fits perfectly in 256 entries
- Utilization: 66.6% (good balance, no waste)

**OUTPUT_SCALE = 16384:**
- 1/√var_hw range: [0.00678, 0.00948]
- Scaled range: [111, 155] → fits in 8-bit [0, 255]
- Utilization: 60.9% (no overflow, excellent precision)

### Error Sources
| Source | Contribution |
|--------|-------------|
| LUT quantization (8-bit) | ~0.3% |
| Index discretization | <0.1% |
| **Total LUT error** | **0.288%** |

Note: Total system error (including dynamic compression in Stage 1) is ~1.5-2.5%, which is excellent for hardware.

---

## Project Structure

```
implementation/
├── README_LUT.md                    ← This file
│
├── scripts/
│   ├── analyze_var_hw_distribution.py   (Step 1: Data analysis)
│   ├── generate_inv_sqrt_lut.py         (Step 2: LUT generation)
│   └── grid_search_lut_params.py        (Step 3: Validation)
│
├── data/lut/
│   ├── lut_inv_sqrt.txt                 (The LUT - 256 values)
│   ├── var_hw_analysis.txt              (Statistical data)
│   ├── grid_search_results.txt          (Detailed comparison)
│   ├── lut_grid_search_heatmap.png      (MSE heatmap)
│   └── lut_grid_search_rel_error.png    (Relative error heatmap)
│
└── src/
    ├── utils.h                          (LUT definitions)
    └── utils.c                          (LUT loading & lookup)
```

---

## Visualizations

The grid search generates two heatmaps showing parameter performance:

**MSE Heatmap:**
Green = low error (good), Red = high error (bad), Blue ★ = optimal configuration

**Relative Error Heatmap:**
Same color scheme showing percentage errors

Both visualizations clearly show that (128, 16384) is the optimal choice.

---

## Hardware Benefits

- **Memory:** Only 256 bytes (fits in on-chip SRAM)
- **Speed:** Single-cycle lookup (vs. 15-30 cycles for sqrt + division)
- **Power:** No floating-point arithmetic needed
- **Accuracy:** <0.3% error is excellent for hardware LayerNorm

---

## For Presentation

**To demonstrate the complete workflow:**
```bash
python scripts/analyze_var_hw_distribution.py
python scripts/generate_inv_sqrt_lut.py
python scripts/grid_search_lut_params.py
```

Then show:
1. The generated heatmaps (visual proof of optimization)
2. The TOP 5 results (our choice is best)
3. The accuracy metrics (0.288% error)

**Key talking points:**
- Data-driven approach (analyzed 197 real vectors)
- Systematic optimization (tested 25 parameter combinations)
- Mathematically optimal (best MSE among all configs)
- Hardware-efficient (256 bytes, <1 cycle access)

---

## Author

Segev
Electrical Engineering, Semester 7
Project: Hardware-Accelerated LayerNorm
Date: January 2026

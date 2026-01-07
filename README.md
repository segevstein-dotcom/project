# SOLE LayerNorm Implementation

**Hardware-Software Co-design of LayerNorm for Efficient Transformer Inference**

This project implements Stage 1 (statistics calculation) of the SOLE (Software-Hardware Co-design of Softmax and LayerNorm) algorithm in C, optimized for hardware acceleration.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Algorithm Overview](#algorithm-overview)
- [Validation](#validation)
- [Results](#results)
- [Documentation](#documentation)

---

## 🎯 Overview

SOLE is a hardware-software co-design approach for efficient LayerNorm and Softmax operations in Transformer models. This implementation focuses on:

- **8-bit Quantization** with PTF (Power-of-Two Factor) per-channel scaling
- **Dynamic Range Compression** (8-bit → 4-bit) for hardware efficiency
- **Square Lookup Table** (16 entries) for fast variance computation
- **Fixed-point Arithmetic** optimized for hardware implementation

### What This Project Does

1. **Generates quantized data** from DeiT-small Transformer model
2. **Implements SOLE Stage 1** in C (mean and variance calculation)
3. **Validates results** against ground truth with comprehensive error analysis
4. **Isolates error sources** (quantization vs algorithm approximations)

---

## ✨ Features

- ✅ **PTF Sequential Quantization** - Per-channel alpha factors for optimal bit utilization
- ✅ **Hardware-Optimized Arithmetic** - Dynamic compression + Square LUT
- ✅ **Comprehensive Validation** - Multiple validation approaches
- ✅ **Detailed Tracing** - Debug individual vectors step-by-step
- ✅ **Well-Organized Structure** - Clean separation of concerns

---

## 📁 Project Structure

```
implementation/
│
├── README.md                    # This file - Project overview
│
├── src/                         # C Implementation
│   ├── main.c                   # SOLE Stage 1 implementation
│   ├── utils.c                  # Data loading utilities
│   ├── def.h                    # Constants and definitions
│   ├── utils.h                  # Function headers
│   ├── build.bat                # Windows build script
│   ├── Makefile                 # Linux/Mac build
│   └── layernorm_test.exe       # Compiled binary
│
├── scripts/                     # Data Generation
│   ├── README.md                # Scripts documentation
│   └── collect_real_data.py     # Generate quantized data from DeiT model
│
├── validation/                  # Validation & Analysis
│   ├── README.md                # Validation documentation
│   ├── validate_statistics.py   # Main validation (total error)
│   ├── compare_c_vs_dequantized.py  # Algorithm error only
│   ├── trace_vector.py          # Debug individual vectors
│   ├── validation_report.txt    # Total error report
│   └── dequantized_comparison.txt   # Algorithm error report
│
├── experiments/                 # Historical Experiments
│   └── quantization/            # Quantization experiments archive
│
└── data/                        # Generated Data
    └── quantized/
        ├── raw_input_vectors/   # Original float32 values (ground truth)
        ├── quantized_vectors/   # PTF-quantized uint8 values (vector_XXX.txt)
        ├── alpha_factors.txt    # Per-channel scaling factors
        ├── global_params.txt    # Global S, ZP, dimensions
        ├── layernorm_weights.txt    # Gamma & Beta (float)
        ├── gamma_quantized.txt  # Quantized gamma weights
        ├── beta_quantized.txt   # Quantized beta weights
        ├── golden_ref_vec000.txt    # Reference output
        └── final_report.txt     # C implementation results
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with packages: `torch`, `transformers`, `numpy`, `PIL`
- **C Compiler**: GCC (Linux/Mac) or MinGW (Windows)
- **Git** (optional, for version control)

### Step-by-Step Workflow

#### 1. Generate Quantized Data

```bash
# From implementation/ directory
python scripts/collect_real_data.py
```

**What it does:**
- Loads DeiT-small Transformer model
- Extracts LayerNorm input vectors (384 channels × 197 vectors)
- Performs PTF quantization with global S/ZP and per-channel alpha
- Generates all data files in `data/quantized/`

#### 2. Compile and Run C Implementation

```bash
# Windows
cd src
build.bat
./layernorm_test.exe
cd ..

# Linux/Mac
cd src
make
./layernorm_test
cd ..
```

**What it does:**
- Loads quantized vectors and parameters
- Computes mean and variance using SOLE algorithm
- Applies dynamic compression (8→4 bit)
- Uses square LUT for variance calculation
- Saves results to `data/quantized/final_report.txt`

#### 3. Validate Results

```bash
# Main validation (quantization + algorithm error)
python validation/validate_statistics.py

# Algorithm-only validation (isolates algorithm error)
python validation/compare_c_vs_dequantized.py

# Debug specific vector (optional)
python validation/trace_vector.py 0
```

---

## 🧮 Algorithm Overview

### SOLE Stage 1: Statistics Calculation

**Goal:** Compute mean (μ) and variance (σ²) efficiently in hardware.

```
Input:  Quantized vector X_q [0-255], alpha factors α, global S, ZP
Output: Mean μ, Variance σ² (in real domain)
```

### Pipeline Stages

#### 1. **Centering** (Remove Zero Point)
```
X_centered = X_q - ZP
```
- Converts unsigned [0,255] to signed [-128,127]

#### 2. **Dynamic Compression** (8-bit → 4-bit)
```
min_alpha = min(α)
X_compressed = compress(X_centered, min_alpha)
```
- Shift all values by min_alpha
- Dynamic range reduction for hardware efficiency
- Preserves relative relationships between channels

#### 3. **Accumulation** (Fixed-point)
```
Ex  = Σ X_compressed[i]                    # Sum
Ex2 = Σ square_lut[X_compressed[i]]        # Sum of squares
```
- **Square LUT**: 16-entry lookup table [-8,7] for fast squaring
- Accumulates in fixed-point (hardware-friendly)

#### 4. **Statistics Computation** (HW Domain)
```
mean_hw = Ex / N
var_hw  = (Ex2 / N) - mean_hw²
```

#### 5. **Scale to Real Domain**
```
full_scale = 2^min_alpha × S
mean_real  = mean_hw × full_scale
var_real   = var_hw × full_scale²        # Var(aX) = a²Var(X)
```

### Quantization: PTF (Power-of-Two Factor)

**Sequential Quantization (SOLE Paper Convention):**

```python
# Step 1: Normalize to [-127, +127]
S = max(|X_real|) / 127
X_norm = X_real / S

# Step 2: Stretch underutilized channels (α ≤ 0)
X_stretched = X_norm / 2^α    # When α < 0, this multiplies (stretching)

# Step 3: Shift to [0, 255]
X_int = X_stretched + ZP      # ZP = 128
```

**Per-Channel Alpha Factors:**
- Channels with low utilization get negative α (e.g., α=-2)
- This stretches them to use more bits
- Channels using full range get α=0

---

## ✅ Validation

### Two Validation Approaches

#### 1. **Total Error** (`validate_statistics.py`)
- **Compares:** C implementation vs original raw float inputs
- **Shows:** Quantization error + Algorithm error (combined)
- **Use for:** Overall system validation

**Current Results:**
```
Mean Error:     7.33%  (average)
Variance Error: 10.40% (average)
Std Error:      5.36%  (average)
```

**Error Sources:**
1. Quantization loss (float32 → uint8 → float32)
2. Dynamic compression (8-bit → 4-bit)
3. Square LUT approximation (16 entries)
4. Fixed-point arithmetic

#### 2. **Algorithm Error Only** (`compare_c_vs_dequantized.py`)
- **Compares:** C implementation vs dequantized quantized values
- **Shows:** Algorithm error only (excludes quantization)
- **Use for:** Understanding algorithm approximations

**Current Results:**
```
Mean Error:     0.004%  (essentially perfect!)
Variance Error: 10.20%  (from compression + LUT)
Std Error:      5.24%   (from compression + LUT)
```

**Key Finding:**
- Mean calculation has **NO algorithm error**
- All mean error comes from quantization, not the algorithm
- Variance/std errors are from dynamic compression and square LUT

---

## 📊 Results Summary

| Metric | Total Error | Algorithm Error |
|--------|-------------|-----------------|
| **Mean** | 7.33% | **0.004%** ✨ |
| **Variance** | 10.40% | 10.20% |
| **Std Dev** | 5.36% | 5.24% |

### Insights

✅ **Mean calculation is perfect** - Algorithm introduces near-zero error
✅ **Quantization dominates mean error** - 7.33% total, 0.004% algorithm
✅ **Variance error is algorithmic** - Dynamic compression + Square LUT
✅ **Acceptable trade-off** - ~10% variance error for significant hardware savings

---

## 📚 Documentation

Each folder contains detailed documentation:

- **[scripts/README.md](scripts/README.md)** - Data generation scripts
- **[validation/README.md](validation/README.md)** - Validation methodology and tools
- **[experiments/quantization/README.md](experiments/quantization/README.md)** - Historical experiments

---

## 🔧 Development

### Building from Source

**Windows (MinGW):**
```bash
cd src
gcc -o layernorm_test.exe main.c utils.c -lm -Wall
```

**Linux/Mac:**
```bash
cd src
gcc -o layernorm_test main.c utils.c -lm -Wall
# or
make
```

### Adding New Tests

1. Generate new data: `python scripts/collect_real_data.py`
2. Run C implementation: `cd src && ./layernorm_test.exe && cd ..`
3. Validate: `python validation/validate_statistics.py`

---

## 📖 References

- **SOLE Paper:** "Hardware-Software Co-design of Softmax and LayerNorm for Efficient Transformer Inference"
- **Model:** DeiT-small (facebook/deit-small-patch16-224)
- **Quantization:** 8-bit symmetric with PTF per-channel scaling

---

## 📝 License

This is an academic project for educational purposes.

---

## 👥 Contributors

- **Segev** - Implementation and validation
- **Claude Sonnet 4.5** - Code assistance and documentation

---

## 🎓 Academic Context

Part of Electrical Engineering Semester 7 project
Hebrew: פרוייקט - סמסטר ז', הנדסת חשמל

---

**Last Updated:** January 2026

/*
 * def.h - Data structures for AILayerNorm SOLE implementation
 */

#ifndef SOLE_DEF_H
#define SOLE_DEF_H

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MAX_CHANNELS 384
#define MAX_FILENAME 256
#define DATA_FOLDER "data/quantized"  // Relative to implementation/ directory

/*
 * PTFLayerNormData: Main data structure for SOLE algorithm
 * Contains quantized inputs, PTF parameters, and LayerNorm weights
 */
typedef struct {
    // Quantized input values (0-255)
    uint8_t quantized_values[MAX_CHANNELS];

    // Per-channel metadata
    int8_t alpha_factors[MAX_CHANNELS];   // PTF exponent (SIGNED! Can be negative)
    uint16_t zero_points[MAX_CHANNELS];   // Per-channel zero-points (can exceed 255)

    // Global SOLE parameters
    float global_s;   // Global scale factor
    int global_zp;    // Global zero-point

    // LayerNorm affine parameters (from PyTorch model)
    float gamma[MAX_CHANNELS];
    float beta[MAX_CHANNELS];

    int num_channels;  // Vector size (384 for DeiT)
    int num_vectors;   // Number of test vectors
} PTFLayerNormData;

// Load all layer data from files
int load_all_layer_data(PTFLayerNormData* data, int vector_idx);

#endif // SOLE_DEF_H

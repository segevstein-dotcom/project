// /*
//  * def.h - Data structures for AILayerNorm SOLE implementation
//  */

// #ifndef SOLE_DEF_H
// #define SOLE_DEF_H

// #include <stdint.h>
// #include <stdio.h>
// #include <math.h>
// #include <stdlib.h>

// #define MAX_CHANNELS 384
// #define MAX_FILENAME 256
// #define DATA_FOLDER "data/quantized"  // Relative to implementation/ directory

// /*
//  * PTFLayerNormData: Main data structure for SOLE algorithm
//  * Contains quantized inputs, PTF parameters, and LayerNorm weights
//  */
// typedef struct {
//     // Quantized input values (0-255)
//     uint8_t quantized_values[MAX_CHANNELS];

//     // Per-channel metadata
//     int8_t alpha_factors[MAX_CHANNELS];   // PTF exponent (SIGNED! Can be negative)
//     uint16_t zero_points[MAX_CHANNELS];   // Per-channel zero-points (can exceed 255)

//     // Global SOLE parameters
//     float global_s;   // Global scale factor
//     int global_zp;    // Global zero-point

//     // LayerNorm affine parameters (from PyTorch model)
//     float gamma[MAX_CHANNELS];
//     float beta[MAX_CHANNELS];

//     int num_channels;  // Vector size (384 for DeiT)
//     int num_vectors;   // Number of test vectors
// } PTFLayerNormData;

// // Load all layer data from files
// int load_all_layer_data(PTFLayerNormData* data, int vector_idx);



// #endif // SOLE_DEF_H


/*
 * def.h - Data structures for AILayerNorm SOLE implementation
 */

#ifndef SOLE_DEF_H
#define SOLE_DEF_H

#include <stdint.h>

#define MAX_CHANNELS 384
#define MAX_FILENAME 256
#define DATA_FOLDER "data/quantized"

/*
 * PTFLayerNormData
 * Holds all data required for SOLE AILayerNorm execution in C
 */
typedef struct {

    // ----------------------------
    // Quantized input vector
    // ----------------------------
    uint8_t  quantized_values[MAX_CHANNELS];   // X_int
    uint16_t zero_points[MAX_CHANNELS];         // ZP
    int8_t   alpha_factors[MAX_CHANNELS];       // PTF alpha

    // ----------------------------
    // LayerNorm affine parameters (QUANTIZED)
    // ----------------------------
    uint8_t gamma_q[MAX_CHANNELS];
    uint8_t beta_q[MAX_CHANNELS];

    /* Also keep original float affine params (from PyTorch) */
    float gamma[MAX_CHANNELS];
    float beta[MAX_CHANNELS];

    float gamma_scale;
    int   gamma_zp;

    float beta_scale;
    int   beta_zp;

    // ----------------------------
    // Global parameters
    // ----------------------------
    float global_s;
    int   global_zp;

    int num_channels;
    int num_vectors;

} PTFLayerNormData;

// Load all required data for one vector
int load_all_layer_data(PTFLayerNormData* data, int vector_idx);

#endif // SOLE_DEF_H

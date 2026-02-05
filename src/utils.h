/*
 * utils.h - Utility functions for AILayerNorm
 * File I/O, validation, and LUT loading
 */

#ifndef SOLE_UTILS_H
#define SOLE_UTILS_H

#include "def.h"
#include <stdio.h>

// ========================================
// File I/O Functions
// ========================================

/*
 * save_output_with_metadata: Write output to file for validation
 */
void save_output_with_metadata(const float* output, int num_channels,
                                float mean, float variance,
                                const char* output_file);

/*
 * validate_against_golden: Compare C output with Python golden reference
 * Returns 0 on success, -1 on failure
 */
int validate_against_golden(const float* output, int num_channels,
                            float mean, float variance,
                            const char* golden_file,
                            FILE* output_stream);

// ========================================
// LUT Management
// ========================================

// 256-entry LUT for inverse square root: f(x) = 1/sqrt(x)
#define INV_SQRT_LUT_SIZE 256
#define INV_SQRT_INPUT_SCALE 16
#define INV_SQRT_OUTPUT_SCALE 64
extern uint8_t INV_SQRT_LUT[INV_SQRT_LUT_SIZE];

/*
 * load_inv_sqrt_lut: Load inverse sqrt LUT from file
 * Returns 0 on success, -1 on failure
 */
int load_inv_sqrt_lut(void);

/*
 * inv_sqrt_lut: Compute 1/sqrt(x) using LUT
 */
float inv_sqrt_lut(float variance);

#endif // SOLE_UTILS_H

float validate_full_vector_with_dequant(int8_t* Y_c_int8, int n, const PTFLayerNormData* data, const char* golden_path);
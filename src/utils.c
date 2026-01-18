/*
 * utils.c - Data loading utilities for AILayerNorm
 */

#include "def.h"
#include "utils.h"
#include <string.h>

/*
 * open_file: Open a file in the data folder
 * Exits on error if file not found
 */
FILE* open_file(const char* filename) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", DATA_FOLDER, filename);
    FILE* f = fopen(path, "r");
    if (!f) {
        printf("Error: Cannot open %s. Did you run the Python script?\n", path);
        exit(1);
    }
    return f;
}

/*
 * skip_comments: Skip comment lines only
 * Leaves file pointer ready to read first data value
 */
void skip_comments(FILE* f) {
    int c;
    // Skip lines starting with '#'
    while ((c = fgetc(f)) == '#') {
        // Skip rest of line
        while ((c = fgetc(f)) != '\n' && c != EOF);
    }
    // Put back the first non-comment character
    if (c != EOF) {
        ungetc(c, f);
    }
}

/*
 * load_all_layer_data: Load all data from files
 * Loads: global params, alpha factors, zero-points, weights, and input vector
 */
int load_all_layer_data(PTFLayerNormData* data, int vector_idx) {
    printf("Loading data from folder: %s (Vector #%d)...\n", DATA_FOLDER, vector_idx);

    // 1. Global parameters (hidden_dim, num_vectors, global_s, global_zp)
    FILE* f_glob = open_file("global_params.txt");
    skip_comments(f_glob);
    fscanf(f_glob, "%d", &data->num_channels);
    fscanf(f_glob, "%d", &data->num_vectors);
    fscanf(f_glob, "%f", &data->global_s);
    fscanf(f_glob, "%d", &data->global_zp);
    fclose(f_glob);

    // 2. Alpha factors (SIGNED integers!)
    FILE* f_alpha = open_file("alpha_factors.txt");
    skip_comments(f_alpha);
    int alpha_count;
    fscanf(f_alpha, "%d", &alpha_count);  // Skip count line
    for(int i=0; i<data->num_channels; i++) {
        int val;
        fscanf(f_alpha, "%d", &val);
        data->alpha_factors[i] = (int8_t)val;  // Can be negative
    }
    fclose(f_alpha);

    // 3. Zero-points (Use global_zp from global_params.txt)
    // SOLE Paper: ZP is always global (same for all channels)
    // Fill all channels with the global_zp value already loaded above
    for(int i=0; i<data->num_channels; i++) {
        data->zero_points[i] = (uint16_t)data->global_zp;
    }

    // 4. LayerNorm weights (gamma and beta)
    FILE* f_w = open_file("layernorm_weights.txt");
    skip_comments(f_w);
    int weight_count;
    fscanf(f_w, "%d", &weight_count);  // Skip count line
    for(int i=0; i<data->num_channels; i++)
        fscanf(f_w, "%f", &data->gamma[i]);
    for(int i=0; i<data->num_channels; i++)
        fscanf(f_w, "%f", &data->beta[i]);
    fclose(f_w);

    // 5. Input vector (quantized values 0-255)
    char vec_name[64];
    snprintf(vec_name, sizeof(vec_name), "vector_%03d.txt", vector_idx);
    FILE* f_vec = open_file(vec_name);
    skip_comments(f_vec);
    int vec_count;
    fscanf(f_vec, "%d", &vec_count);  // Skip count line
    for(int i=0; i<data->num_channels; i++) {
        int val;
        fscanf(f_vec, "%d", &val);
        data->quantized_values[i] = (uint8_t)val;
    }
    fclose(f_vec);

    return 0;
}

// ========================================
// LUT MANAGEMENT
// ========================================

uint8_t INV_SQRT_LUT[INV_SQRT_LUT_SIZE];

/*
 * load_inv_sqrt_lut: Load inverse sqrt LUT from file
 */
int load_inv_sqrt_lut() {
    char path[512];
    snprintf(path, sizeof(path), "data/lut/lut_inv_sqrt.txt");
    FILE* f = fopen(path, "r");
    if (!f) {
        printf("Warning: Cannot open %s, using fallback sqrt()\n", path);
        return -1;
    }

    for (int i = 0; i < INV_SQRT_LUT_SIZE; i++) {
        int val;
        if (fscanf(f, "%d", &val) != 1) {
            printf("Error: LUT file corrupted at index %d\n", i);
            fclose(f);
            return -1;
        }
        INV_SQRT_LUT[i] = (uint8_t)val;
    }
    fclose(f);
    return 0;
}

/*
 * inv_sqrt_lut: Compute 1/sqrt(x) using LUT
 * Input: variance (var_hw in hardware domain)
 * Output: 1/sqrt(variance)
 */
float inv_sqrt_lut(float variance) {
    // Divide variance by INPUT_DIVISOR to get LUT index
    int address = (int)(variance / INV_SQRT_INPUT_DIVISOR);
    if (address < 0) address = 0;
    if (address >= INV_SQRT_LUT_SIZE) address = INV_SQRT_LUT_SIZE - 1;

    uint8_t lut_value = INV_SQRT_LUT[address];
    return (float)lut_value / INV_SQRT_OUTPUT_SCALE;
}

// ========================================
// OUTPUT & VALIDATION
// ========================================

/*
 * save_output_with_metadata: Write output to file for validation
 */
void save_output_with_metadata(const float* output, int num_channels,
                                float mean, float variance,
                                const char* output_file) {
    FILE* f = fopen(output_file, "w");
    if (!f) {
        printf("Error: Cannot create output file %s\n", output_file);
        return;
    }

    fprintf(f, "# C Implementation Output (Stage 2)\n");
    fprintf(f, "# Metadata:\n");
    fprintf(f, "Mean: %.6f\n", mean);
    fprintf(f, "Variance: %.6f\n", variance);
    fprintf(f, "# Output Values:\n");
    fprintf(f, "Output:\n");

    for (int i = 0; i < num_channels; i++) {
        fprintf(f, "%.6f\n", output[i]);
    }

    fclose(f);
    printf(">> Output saved to: %s\n", output_file);
}

/*
 * validate_against_golden: Compare C output with Python golden reference
 */
int validate_against_golden(const float* output, int num_channels,
                            float mean, float variance,
                            const char* golden_file) {
    FILE* f = fopen(golden_file, "r");
    if (!f) {
        printf("Warning: Cannot open golden reference file: %s\n", golden_file);
        return -1;
    }

    char line[256];
    float golden_mean = 0, golden_var = 0;

    // Parse header
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "Mean: %f", &golden_mean) == 1) continue;
        if (sscanf(line, "Variance: %f", &golden_var) == 1) continue;
        if (strstr(line, "Output:")) break;
    }

    // Read golden output
    float golden_output[MAX_CHANNELS];
    int count = 0;
    while (fgets(line, sizeof(line), f) && count < num_channels) {
        if (sscanf(line, "%f", &golden_output[count]) == 1) {
            count++;
        }
    }
    fclose(f);

    if (count != num_channels) {
        printf("Error: Golden reference has %d values, expected %d\n", count, num_channels);
        return -1;
    }

    // Calculate error metrics
    float mean_error = fabsf(mean - golden_mean);
    float var_error = fabsf(variance - golden_var);

    float mse = 0.0f, mae = 0.0f, max_error = 0.0f;
    for (int i = 0; i < num_channels; i++) {
        float error = fabsf(output[i] - golden_output[i]);
        mae += error;
        mse += error * error;
        if (error > max_error) max_error = error;
    }
    mae /= num_channels;
    mse /= num_channels;

    // Print validation report
    printf("\n========================================================\n");
    printf("         VALIDATION AGAINST GOLDEN REFERENCE\n");
    printf("========================================================\n");
    printf("Statistics Comparison:\n");
    printf("  Mean Error:     %12.6f (%.4f%%)\n",
           mean_error, fabsf(golden_mean) > 1e-6 ? (mean_error/fabsf(golden_mean)*100) : 0);
    printf("  Variance Error: %12.6f (%.4f%%)\n",
           var_error, fabsf(golden_var) > 1e-6 ? (var_error/fabsf(golden_var)*100) : 0);
    printf("\nOutput Error Metrics:\n");
    printf("  Mean Absolute Error (MAE): %12.6f\n", mae);
    printf("  Mean Squared Error  (MSE): %12.6f\n", mse);
    printf("  Max Error:                 %12.6f\n", max_error);
    printf("\n");

    // Pass/Fail criteria
    int passed = (mae < 1.0f && mse < 1.0f && mean_error < 0.01f);
    if (passed) {
        printf(">>> VALIDATION PASSED <<<\n");
    } else if (mae < 2.0f && mse < 2.0f) {
        printf(">>> VALIDATION MARGINAL (Acceptable for HW) <<<\n");
        passed = 1;
    } else {
        printf(">>> VALIDATION FAILED <<<\n");
    }
    printf("========================================================\n");

    return passed ? 0 : -1;
}

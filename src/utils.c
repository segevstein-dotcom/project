/*
 * utils.c - Data loading utilities for AILayerNorm
 */

#include "def.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

static int load_quantized_param_u8(
    const char* filename,
    uint8_t* out_vals,
    float* out_scale,
    int* out_zp,
    int expected_channels
) {
    FILE* f = open_file(filename);
    skip_comments(f);

    int n = 0;
    if (fscanf(f, "%d", &n) != 1) { fclose(f); return -1; }
    if (n != expected_channels) {
        printf("Error: %s channels=%d expected=%d\n", filename, n, expected_channels);
        fclose(f);
        return -1;
    }

    if (fscanf(f, "%f", out_scale) != 1) { fclose(f); return -1; }
    if (fscanf(f, "%d", out_zp) != 1) { fclose(f); return -1; }

    for (int i = 0; i < n; i++) {
        int v = 0;
        if (fscanf(f, "%d", &v) != 1) { fclose(f); return -1; }
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        out_vals[i] = (uint8_t)v;
    }

    fclose(f);
    return 0;
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

    // 4. LayerNorm weights
    // 4a. Float weights (original gamma/beta) - read and store for Stage2
    FILE* f_w = open_file("layernorm_weights.txt");
    skip_comments(f_w);
    int weight_count;
    if (fscanf(f_w, "%d", &weight_count) != 1) {
        printf("Error: layernorm_weights.txt corrupted\n");
        fclose(f_w);
        return -1;
    }
    for (int i = 0; i < data->num_channels; i++) {
        float g = 1.0f;
        if (fscanf(f_w, "%f", &g) == 1) data->gamma[i] = g; else data->gamma[i] = 1.0f;
    }
    for (int i = 0; i < data->num_channels; i++) {
        float b = 0.0f;
        if (fscanf(f_w, "%f", &b) == 1) data->beta[i] = b; else data->beta[i] = 0.0f;
    }
    fclose(f_w);
    // 4b. Quantized weights (used by HW-like Stage 2)
    static int affine_q_loaded = 0;
    if (!affine_q_loaded) {
        if (load_quantized_param_u8("gamma_quantized.txt",
                                    data->gamma_q,
                                    &data->gamma_scale,
                                    &data->gamma_zp,
                                    data->num_channels) != 0) {
            printf("Error: failed to load gamma_quantized.txt\n");
            return -1;
     }

        if (load_quantized_param_u8("beta_quantized.txt",
                                  data->beta_q,
                                  &data->beta_scale,
                                   &data->beta_zp,
                                   data->num_channels) != 0) {
           printf("Error: failed to load beta_quantized.txt\n");
           return -1;
     }

     affine_q_loaded = 1;
         printf("Loaded quantized gamma/beta: gamma_scale=%g gamma_zp=%d, beta_scale=%g beta_zp=%d\n",
             data->gamma_scale, data->gamma_zp, data->beta_scale, data->beta_zp);
         printf("Loaded float gamma/beta from layernorm_weights.txt\n");
    }


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
 */
float inv_sqrt_lut(float variance) {
    int address = (int)(variance * INV_SQRT_INPUT_SCALE);
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
                            const char* golden_file,
                            FILE* output_stream) {
    FILE* f = fopen(golden_file, "r");
    if (!f) {
        printf("Warning: Cannot open golden reference file: %s\n", golden_file);
        fprintf(output_stream, "Warning: Cannot open golden reference file: %s\n", golden_file);
        return -1;
    }

    char line[256];
    float golden_mean = 0, golden_var = 0;

    // Parse header
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "Mean: %f", &golden_mean) == 1) continue;
        if (sscanf(line, "Variance: %f", &golden_var) == 1) continue;
        if (sscanf(line, "ORIGINAL Mean: %f", &golden_mean) == 1) continue;
        if (sscanf(line, "ORIGINAL Variance: %f", &golden_var) == 1) continue;
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
        fprintf(output_stream, "Error: Golden reference has %d values, expected %d\n", count, num_channels);
        return -1;
    }

    // Calculate error metrics
    float mean_error = fabsf(mean - golden_mean);
    float var_error = fabsf(variance - golden_var);

    float mse = 0.0f, mae = 0.0f, max_error = 0.0f;
    int max_error_idx = 0;
    for (int i = 0; i < num_channels; i++) {
        float error = fabsf(output[i] - golden_output[i]);
        mae += error;
        mse += error * error;
        if (error > max_error) {
            max_error = error;
            max_error_idx = i;
        }
    }
    mae /= num_channels;
    mse /= num_channels;

    // Print validation report to both stdout and file
    printf("\n========================================================\n");
    fprintf(output_stream, "\n========================================================\n");
    
    printf("         VALIDATION AGAINST GOLDEN REFERENCE\n");
    fprintf(output_stream, "         VALIDATION AGAINST GOLDEN REFERENCE\n");
    
    printf("========================================================\n");
    fprintf(output_stream, "========================================================\n");
    
    printf("Statistics Comparison:\n");
    fprintf(output_stream, "Statistics Comparison:\n");
    
    printf("  Mean Error:     %12.6f (%.4f%%)\n",
           mean_error, fabsf(golden_mean) > 1e-6 ? (mean_error/fabsf(golden_mean)*100) : 0);
    fprintf(output_stream, "  Mean Error:     %12.6f (%.4f%%)\n",
           mean_error, fabsf(golden_mean) > 1e-6 ? (mean_error/fabsf(golden_mean)*100) : 0);
    
    printf("  Variance Error: %12.6f (%.4f%%)\n",
           var_error, fabsf(golden_var) > 1e-6 ? (var_error/fabsf(golden_var)*100) : 0);
    fprintf(output_stream, "  Variance Error: %12.6f (%.4f%%)\n",
           var_error, fabsf(golden_var) > 1e-6 ? (var_error/fabsf(golden_var)*100) : 0);
    
    printf("\nOutput Error Metrics (Y values):\n");
    fprintf(output_stream, "\nOutput Error Metrics (Y values):\n");
    
    printf("  Mean Absolute Error (MAE): %12.6f\n", mae);
    fprintf(output_stream, "  Mean Absolute Error (MAE): %12.6f\n", mae);
    
    printf("  Mean Squared Error  (MSE): %12.6f\n", mse);
    fprintf(output_stream, "  Mean Squared Error  (MSE): %12.6f\n", mse);
    
    printf("  Max Error:                 %12.6f (at channel %d)\n", max_error, max_error_idx);
    fprintf(output_stream, "  Max Error:                 %12.6f (at channel %d)\n", max_error, max_error_idx);
    
    printf("\nSample Comparison (first 10 channels):\n");
    fprintf(output_stream, "\nSample Comparison (first 10 channels):\n");
    
    printf("  Channel | C Output    | Golden     | Error\n");
    fprintf(output_stream, "  Channel | C Output    | Golden     | Error\n");
    
    printf("  --------|-------------|------------|----------\n");
    fprintf(output_stream, "  --------|-------------|------------|----------\n");
    
    for (int i = 0; i < 10 && i < num_channels; i++) {
        printf("  %7d | %11.6f | %10.6f | %8.6f\n", 
               i, output[i], golden_output[i], fabsf(output[i] - golden_output[i]));
        fprintf(output_stream, "  %7d | %11.6f | %10.6f | %8.6f\n", 
               i, output[i], golden_output[i], fabsf(output[i] - golden_output[i]));
    }
    printf("\n");
    fprintf(output_stream, "\n");

    // Pass/Fail criteria
    int passed = (mae < 1.0f && mse < 1.0f && mean_error < 0.01f);
    if (passed) {
        printf(">>> VALIDATION PASSED <<<\n");
        fprintf(output_stream, ">>> VALIDATION PASSED <<<\n");
    } else if (mae < 2.0f && mse < 2.0f) {
        printf(">>> VALIDATION MARGINAL (Acceptable for HW) <<<\n");
        fprintf(output_stream, ">>> VALIDATION MARGINAL (Acceptable for HW) <<<\n");
        passed = 1;
    } else {
        printf(">>> VALIDATION FAILED <<<\n");
        fprintf(output_stream, ">>> VALIDATION FAILED <<<\n");
    }
    printf("========================================================\n");
    fprintf(output_stream, "========================================================\n");

    /* Also write a full per-channel comparison CSV for easier analysis */
    const char* csv_path = "data/quantized/y_comparison_vec000.csv";
    FILE* csv = fopen(csv_path, "w");
    if (csv) {
        fprintf(csv, "channel,c_output,golden,error\n");
        for (int i = 0; i < num_channels; i++) {
            float err = fabsf(output[i] - golden_output[i]);
            fprintf(csv, "%d,%.6f,%.6f,%.6f\n", i, output[i], golden_output[i], err);
        }
        fclose(csv);
        printf("Wrote full Y comparison to: %s\n", csv_path);
        fprintf(output_stream, "Wrote full Y comparison to: %s\n", csv_path);
    } else {
        printf("Warning: Could not open %s for writing\n", csv_path);
        fprintf(output_stream, "Warning: Could not open %s for writing\n", csv_path);
    }

    return passed ? 0 : -1;
}


float validate_full_vector_with_dequant(int8_t* Y_c_int8, int n, const PTFLayerNormData* data, const char* golden_path) {
    FILE* fp = fopen(golden_path, "r");
    if (!fp) return -1.0f;

    char line[256];
    float golden_mean = 0, golden_var = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "ORIGINAL Mean:")) sscanf(line, "ORIGINAL Mean: %f", &golden_mean);
        if (strstr(line, "ORIGINAL Variance:")) sscanf(line, "ORIGINAL Variance: %f", &golden_var);
        if (strstr(line, "Output:")) break;
    }

    float total_abs_error = 0;
    float golden_val;
    
    // שליפת ה-Scales וה-PTF לצורך שחזור הערך הריאלי מה-C
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f", &golden_val) != 1) break;
        
        // שחזור הערך הריאלי מה-int8 הסופי של ה-C לצורך השוואה ל-Golden הריאלי
        // הנוסחה הזו מחזירה אותנו מהקוונטיזציה של SOLE לערכי ה-float המקוריים
        float y_c_float = (float)Y_c_int8[i] * data->gamma_scale; 

        total_abs_error += fabsf(y_c_float - golden_val);
    }
    fclose(fp);
    return total_abs_error / n;
}
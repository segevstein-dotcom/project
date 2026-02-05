#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "def.h"
#include "utils.h"

// ========================================
// GLOBAL CONSTANTS
// ========================================

const uint16_t SQUARE_LUT[16] = {
    0, 1, 4, 9, 16, 25, 36, 49,
    64, 81, 100, 121, 144, 169, 196, 225
};

void dynamic_compress(uint8_t x, uint8_t* compressed, uint8_t* shift) {
    if (x >= 64) { 
        *compressed = x >> 4; 
        *shift = 1;
    } else { 
        *compressed = x >> 2; 
        *shift = 0;
    }
    if (*compressed > 15) *compressed = 15;
}

int8_t find_min_alpha(const int8_t* alphas, int num_channels) {
    int8_t min_alpha = 127;
    for (int i = 0; i < num_channels; i++) {
        if (alphas[i] < min_alpha) min_alpha = alphas[i];
    }
    return min_alpha;
}

// ========================================
// CORE ALGORITHM
// ========================================

void stage1(const PTFLayerNormData* data, int64_t* Ex, int64_t* Ex2, int8_t min_alpha) {
    *Ex = 0; *Ex2 = 0;
    for (int i = 0; i < data->num_channels; i++) {
        int16_t xi = (int16_t)data->quantized_values[i] - (int16_t)data->zero_points[i];
        uint8_t abs_xi = (xi < 0) ? -xi : xi;
        
        uint8_t compressed, s;
        dynamic_compress(abs_xi, &compressed, &s);
        uint32_t xc_sq = (uint32_t)SQUARE_LUT[compressed] << (4 * s);

        int relative_shift = data->alpha_factors[i] - min_alpha;
        *Ex += (int64_t)xi << relative_shift;
        *Ex2 += (int64_t)xc_sq << (2 * relative_shift);
    }
}

const uint32_t STD_INV_LUT[16] = {
    1448, 1448, 1024, 836, 724, 647, 591, 547,
    512,  482,  458,  436, 418, 401, 387, 374
};

void stage2(const PTFLayerNormData* data, int64_t Ex, int64_t Ex2, int8_t min_alpha, int8_t* Y_out_int8) {
    int C = data->num_channels;
    int64_t mu = Ex / C; 
    int64_t mean_sq = (Ex2 << 4) / C; 
    int64_t var = mean_sq - (mu * mu);
    if (var < 0) var = 0;

    // --- Alignment Step ---
    // We use shift 11 to spread our variance (~8000) over 32 entries (0-31)
    uint8_t lut_index = (uint8_t)(var >> 11); 
    if (lut_index > 15) lut_index = 15;
    
    // Pull pre-scaled value from Calibrated LUT
    int32_t std_inv_q16 = (int32_t)STD_INV_LUT[lut_index];

    for (int i = 0; i < C; i++) {
        int32_t g_q = (int32_t)data->gamma_q[i] - data->gamma_zp;
        int32_t b_q = (int32_t)data->beta_q[i] - data->beta_zp;
    
        int32_t xi_ptf = (int32_t)((int16_t)data->quantized_values[i] - data->zero_points[i]) << (data->alpha_factors[i] - min_alpha);
        
        // Final Affine: (X-mu) * std_inv * gamma + beta
        // The shift 16 removes the Q16 scaling
        int32_t y = (int32_t)(((int64_t)(xi_ptf - (int32_t)mu) * std_inv_q16 * g_q) >> 16) + b_q;
        
        if (y > 127) y = 127; else if (y < -128) y = -128;
        Y_out_int8[i] = (int8_t)y;
    }
}


int main() {
    PTFLayerNormData data;
    const int NUM_VECTORS = 197;

    float global_total_mae = 0.0f;
    float total_mean_err_pct = 0.0f;
    float total_var_err_pct = 0.0f;
    
    float max_mae = 0.0f;
    int max_mae_vec_idx = -1;

    char report_path[512];
    snprintf(report_path, sizeof(report_path), "%s/final_report.txt", DATA_FOLDER);
    FILE* f = fopen(report_path, "w");
    if (!f) return 1;

    // כותרת טבלה מעודכנת - ללא SNR ועם דגש על השוואת פלט (Y)
    fprintf(f, "| ID   | Mean (Gold) | Mean (C) | Mean Err%% | Var (Gold) | Var (C)  | Var Err%% | Y MAE (%%) |\n");
    fprintf(f, "|------|-------------|----------|-----------|------------|----------|----------|-----------|\n");

    for (int vec_idx = 0; vec_idx < NUM_VECTORS; vec_idx++) {
        if (load_all_layer_data(&data, vec_idx) != 0) break;

        int8_t min_alpha = find_min_alpha(data.alpha_factors, data.num_channels);
        int64_t Ex, Ex2;
        stage1(&data, &Ex, &Ex2, min_alpha);

        // חישוב ערכים ב-C
        float fs = powf(2.0f, (float)min_alpha) * data.global_s;
        float mean_c = ((float)Ex / data.num_channels) * fs;
        float var_c = ((float)(Ex2 << 4) / data.num_channels - powf((float)Ex/data.num_channels, 2)) * (fs * fs);

        int8_t Y_out[MAX_CHANNELS];
        stage2(&data, Ex, Ex2, min_alpha, Y_out);

        // טעינת ערכי Golden מהקובץ
        char golden_path[512];
        snprintf(golden_path, sizeof(golden_path), "%s/golden_ref_vec%03d.txt", DATA_FOLDER, vec_idx);
        float gold_mean = 0, gold_var = 0;
        FILE* fg = fopen(golden_path, "r");
        if (fg) {
            char line[256];
            while (fgets(line, sizeof(line), fg)) {
                if (strstr(line, "ORIGINAL Mean:")) sscanf(line, "ORIGINAL Mean: %f", &gold_mean);
                else if (strstr(line, "ORIGINAL Variance:")) sscanf(line, "ORIGINAL Variance: %f", &gold_var);
            }
            fclose(fg);
        }

        // חישוב שגיאות יחסיות (אחוזים)
        float mean_err = (fabsf(gold_mean) > 1e-6) ? fabsf((gold_mean - mean_c) / gold_mean) * 100.0f : 0.0f;
        float var_err = (gold_var > 1e-6) ? fabsf((gold_var - var_c) / gold_var) * 100.0f : 0.0f;
        
        // MAE מייצג את ההפרש הממוצע בין Y_c ל-Y_golden
        float mae = validate_full_vector_with_dequant(Y_out, data.num_channels, &data, golden_path);

        if (mae > max_mae) {
            max_mae = mae;
            max_mae_vec_idx = vec_idx;
        }

        global_total_mae += mae;
        total_mean_err_pct += mean_err;
        total_var_err_pct += var_err;

        // הדפסת שורה לדו"ח
        fprintf(f, "| %-4d | %11.4f | %8.4f | %8.2f%% | %10.4f | %8.4f | %7.2f%% | %8.2f%% |\n", 
                vec_idx, gold_mean, mean_c, mean_err, gold_var, var_c, var_err, mae * 100.0f);
    }

    // סיכום סופי ברור
    fprintf(f, "\n==================================================\n");
    fprintf(f, "        FINAL PERFORMANCE SUMMARY (SOLE)\n");
    fprintf(f, "==================================================\n");
    fprintf(f, "Total Vectors Processed     : %d\n", NUM_VECTORS);
    fprintf(f, "Avg. Statistics Error (Mean): %.2f%%\n", total_mean_err_pct / NUM_VECTORS);
    fprintf(f, "Avg. Statistics Error (Var) : %.2f%%\n", total_var_err_pct / NUM_VECTORS);
    fprintf(f, "--------------------------------------------------\n");
    fprintf(f, "OVERALL ALGORITHM PRECISION (MAE): %.4f%%\n", (global_total_mae / NUM_VECTORS) * 100.0f);
    fprintf(f, "==================================================\n");

    fclose(f);
    printf("Comprehensive report generated: final_report.txt\n");
    return 0;
}
/*
 * main.c - SOLE Algorithm Implementation (Global Scale)
 * Generates a readable text report for all vectors.
 */

#include "def.h"
#include "utils.h"
#include <math.h> 
#include <stdio.h>

// ========================================
// GLOBAL CONSTANTS & HELPERS
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

// פונקציה למציאת האלפא המינימלית (עבור Global Scale)
int8_t find_min_alpha(const int8_t* alphas, int num_channels) {
    int8_t min_alpha = 127;
    for (int i = 0; i < num_channels; i++) {
        if (alphas[i] < min_alpha) {
            min_alpha = alphas[i];
        }
    }
    return min_alpha;
}

// ========================================
// STAGE 1 STATISTICS (GLOBAL SCALE)
// ========================================

void stage1_statistics(const PTFLayerNormData* data, int64_t* Ex, int64_t* Ex2, int8_t min_alpha) {
    *Ex = 0;
    *Ex2 = 0;

    for (int i = 0; i < data->num_channels; i++) {
        uint16_t zp = data->zero_points[i];
        uint8_t input_val = data->quantized_values[i];

        // 1. Center
        int16_t xi = (int16_t)input_val - (int16_t)zp;

        // 2. Compress & Square
        uint8_t abs_xi = (xi < 0) ? -xi : xi;
        uint8_t compressed, shift;
        dynamic_compress(abs_xi, &compressed, &shift);
        uint32_t xc_sq = (uint32_t)SQUARE_LUT[compressed] << (4 * shift);

        // 3. Global Scale Logic (Shift Left Only)
        int8_t alpha = data->alpha_factors[i];
        int relative_shift = alpha - min_alpha; // Always >= 0

        int64_t ex_contribution = (int64_t)xi << relative_shift;
        int64_t ex2_contribution = (int64_t)xc_sq << (2 * relative_shift);

        // 4. Accumulate
        *Ex += ex_contribution;
        *Ex2 += ex2_contribution;
    }
}

// ========================================
// MAIN PROGRAM
// ========================================

int main(int argc, char** argv) {
    PTFLayerNormData data;
    const int NUM_VECTORS = 197; // מספר הוקטורים בדאטהסט שלך

    printf("========================================================\n");
    printf("      AILayerNorm - Generating Readable Report\n");
    printf("========================================================\n");

    // 1. יצירת קובץ הטקסט
    char report_path[512];
    snprintf(report_path, sizeof(report_path), "%s/final_report.txt", DATA_FOLDER);
    FILE* f = fopen(report_path, "w");
    
    if (f == NULL) {
        printf("Error: Could not open file for writing: %s\n", report_path);
        return 1;
    }

    // 2. כתיבת כותרת יפה לטבלה בקובץ
    fprintf(f, "========================================================================================\n");
    fprintf(f, "                                 GLOBAL SCALE RESULTS REPORT                            \n");
    fprintf(f, "========================================================================================\n");
    fprintf(f, "| Vec ID | Min Alpha |    Mean (Real)   |  Variance (Real) |    Std (Real)    | Ex (Int) |\n");
    fprintf(f, "|--------|-----------|------------------|------------------|------------------|----------|\n");

    printf("Processing vectors and writing to %s...\n", report_path);

    // 3. לולאה על כל הוקטורים
    for (int vec_idx = 0; vec_idx < NUM_VECTORS; vec_idx++) {
        
        // טעינת וקטור
        if (load_all_layer_data(&data, vec_idx) != 0) {
            break; 
        }

        // --- חישובים ---
        int8_t min_alpha = find_min_alpha(data.alpha_factors, data.num_channels);
        
        int64_t Ex, Ex2;
        stage1_statistics(&data, &Ex, &Ex2, min_alpha);

        // המרה ל-Float (במימד ה-HW המנופח)
        float mean_hw = (float)Ex / data.num_channels;
        float mean_sq_hw = (float)(Ex2 << 4) / data.num_channels;
        float var_hw = mean_sq_hw - (mean_hw * mean_hw);
        if (var_hw < 0) var_hw = 0;

        // המרה חזרה ל-Real Domain (בשביל הדוח האנושי)
        // צריך שני scale factors: 2^min_alpha וגם global_s!
        float alpha_scale = powf(2.0f, (float)min_alpha);
        float full_scale = alpha_scale * data.global_s;  // IMPORTANT: multiply by global_s too!
        float mean_real = mean_hw * full_scale;
        float var_real = var_hw * (full_scale * full_scale);  // variance scales by s²
        float std_real = sqrtf(var_real);

        // --- כתיבה לקובץ הטקסט ---
        // שימוש בפורמט קבוע (%-8d וכו') כדי שהטבלה תצא מיושרת ויפה
        fprintf(f, "| %-6d | %-9d | %16.6f | %16.6f | %16.6f | %-8lld |\n", 
                vec_idx, min_alpha, mean_real, var_real, std_real, Ex);
        
        // הדפסת התקדמות למסך
        if (vec_idx % 20 == 0) {
            printf("  Processed vector %d / %d\n", vec_idx, NUM_VECTORS);
        }
    }

    fprintf(f, "========================================================================================\n");
    fclose(f);

    printf("\nDone! Open the file below to see the results:\n");
    printf(">> %s\n", report_path);

    return 0;
}
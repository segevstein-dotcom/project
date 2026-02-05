#include "def.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// פונקציית עזר לפתיחת קבצים
FILE* open_file(const char* filename) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", DATA_FOLDER, filename);
    FILE* f = fopen(path, "r");
    if (!f) {
        printf("Error: Cannot open %s\n", path);
        exit(1);
    }
    return f;
}

// דילוג על הערות בקבצי הטקסט
void skip_comments(FILE* f) {
    int c;
    while ((c = fgetc(f)) == '#') {
        while ((c = fgetc(f)) != '\n' && c != EOF);
    }
    if (c != EOF) ungetc(c, f);
}

// טעינת פרמטרים קוונטיזיים (Gamma/Beta)
static int load_quantized_param_u8(const char* filename, uint8_t* out_vals, float* out_scale, int* out_zp, int expected_channels) {
    FILE* f = open_file(filename);
    skip_comments(f);
    int n = 0;
    if (fscanf(f, "%d", &n) != 1 || n != expected_channels) { fclose(f); return -1; }
    if (fscanf(f, "%f", out_scale) != 1) { fclose(f); return -1; }
    if (fscanf(f, "%d", out_zp) != 1) { fclose(f); return -1; }

    for (int i = 0; i < n; i++) {
        int v;
        if (fscanf(f, "%d", &v) != 1) { fclose(f); return -1; }
        out_vals[i] = (uint8_t)fmin(fmax(v, 0), 255);
    }
    fclose(f);
    return 0;
}

// הפונקציה המרכזית לטעינת נתונים
int load_all_layer_data(PTFLayerNormData* data, int vector_idx) {
    // 1. פרמטרים גלובליים
    FILE* f_glob = open_file("global_params.txt");
    skip_comments(f_glob);
    fscanf(f_glob, "%d %d %f %d", &data->num_channels, &data->num_vectors, &data->global_s, &data->global_zp);
    fclose(f_glob);

    // 2. Alpha factors
    FILE* f_alpha = open_file("alpha_factors.txt");
    skip_comments(f_alpha);
    int alpha_count; fscanf(f_alpha, "%d", &alpha_count);
    for(int i=0; i<data->num_channels; i++) {
        int val; fscanf(f_alpha, "%d", &val);
        data->alpha_factors[i] = (int8_t)val;
    }
    fclose(f_alpha);

    // 3. הגדרת Zero-points
    for(int i=0; i<data->num_channels; i++) data->zero_points[i] = (uint16_t)data->global_zp;

    // 4. טעינת משקולות קוונטיזיות (טעינה חד פעמית)
    static int affine_q_loaded = 0;
    if (!affine_q_loaded) {
        load_quantized_param_u8("gamma_quantized.txt", data->gamma_q, &data->gamma_scale, &data->gamma_zp, data->num_channels);
        load_quantized_param_u8("beta_quantized.txt", data->beta_q, &data->beta_scale, &data->beta_zp, data->num_channels);
        affine_q_loaded = 1;
    }

    // 5. טעינת הוקטור הנוכחי
    char vec_name[64];
    snprintf(vec_name, sizeof(vec_name), "vector_%03d.txt", vector_idx);
    FILE* f_vec = open_file(vec_name);
    skip_comments(f_vec);
    int vec_count; fscanf(f_vec, "%d", &vec_count);
    for(int i=0; i<data->num_channels; i++) {
        int val; fscanf(f_vec, "%d", &val);
        data->quantized_values[i] = (uint8_t)val;
    }
    fclose(f_vec);
    return 0;
}

// פונקציית ולידציה בודדת ומדויקת
float validate_full_vector_with_dequant(int8_t* Y_c_int8, int n, const PTFLayerNormData* data, const char* golden_path) {
    FILE* fp = fopen(golden_path, "r");
    if (!fp) return -1.0f;

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strstr(line, "Output:")) break;
    }

    float total_abs_error = 0;
    for (int i = 0; i < n; i++) {
        float golden_val;
        if (fscanf(fp, "%f", &golden_val) != 1) break;
        float y_c_float = (float)Y_c_int8[i] * data->gamma_scale; 
        total_abs_error += fabsf(y_c_float - golden_val);
    }
    fclose(fp);
    return total_abs_error / n;
}
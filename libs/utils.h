
#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "structs.h"

float *float_vector(int size) {
    return (float *) malloc(size * sizeof(float));
}

void show_matrix(float *m, int limit, int rows, int cols) {
    if (limit == 0) return;

    int end = (limit > 0 && limit <= rows) ? limit : rows;

    for (int i = 0; i < end; i++) {
        printf("[");
        for (int j = 0; j < cols; j++) {
            printf("%f, ", m[i * cols + j]);
        }
        printf("],\n");
    }
    printf("\nSize: %i x %i\n============\n", end, cols);
}

float **to_batches(float *X, int batch_size, int n, int rows, int columns) {
    float **batches = (float **) malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        int start = i * batch_size;
        int end = (i < n - 1) ? (i + 1) * batch_size : rows;
        batches[i] = (float *) malloc((end - start) * columns * sizeof(float));
        for (int j = start; j < end; j++) {
            for (int k = 0; k < columns; k++) {
                batches[i][(j - start) * columns + k] = X[j * columns + k];
            }
        }
    }

    return batches;
}

char **get_paths(char *dataset) {
    static const char *base_dir = "./datasets/";

    char **paths = (char **) malloc(2 * sizeof(char *));
    paths[0] = (char *) malloc(512 * sizeof(char));
    paths[1] = (char *) malloc(512 * sizeof(char));

    sprintf(paths[0], "%s%s/train.csv", base_dir, dataset);
    sprintf(paths[1], "%s%s/test.csv", base_dir, dataset);

    return paths;
}

void write_metrics(int epoch, float rmse, float loss, float acc, char *dataset) {
    static const char *base_dir = "./datasets/";
    char *file_path = (char *) malloc(512 * sizeof(char));
    sprintf(file_path, "%s%s/metrics.csv", base_dir, dataset);

    FILE *arquivo = fopen(file_path, "w");

    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo!\n");
        return;
    }
    
    fprintf(arquivo, "%i,%f,%f,%f\n", epoch, rmse, loss, acc);
    fclose(arquivo);
    free(file_path);
}

void write_cmp(float *y, float *output, int n, char *dataset) {
    static const char *base_dir = "./datasets/";
    char *file_path = (char *) malloc(512 * sizeof(char));
    sprintf(file_path, "%s%s/comp.csv", base_dir, dataset);

    FILE *arquivo = fopen(file_path, "w");

    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo!\n");
        return;
    }

    for (int i = 0; i < n; i++) fprintf(arquivo, "%f,%f,%f,%i\n", y[i], output[i], y[i]- output[i], fabsf(y[i] - output[i]) < 0.001);
    fclose(arquivo);
}

void write_pred(float *pred, char *dataset, int rows) {
    static const char *base_dir = "./datasets/";
    char *file_path = (char *) malloc(512 * sizeof(char));
    sprintf(file_path, "%s%s/pred.csv", base_dir, dataset);

    FILE *arquivo = fopen(file_path, "w");

    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo!\n");
        return;
    }

    printf("%i\n", rows);

    for (int i = 0; i < rows; i++) fprintf(arquivo, "%f\n", pred[i]);
    fclose(arquivo);
}

char *load_config(const char *filename, Config *config,
    int *train_size, int *test_size, int *columns) {

    char *dataset;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erro ao abrir o arquivo '%s'\n", filename);
        return NULL;
    }

    char line[256];

    while (fgets(line, sizeof(line), fp)) {
        char *key, *value;

        char *comment = strchr(line, '#');
        if (comment) *comment = '\0';

        key = strtok(line, "=");
        if (!key) continue;

        value = strtok(NULL, "=");
        if (!value) continue;
    
        while (*value == ' ' || *value == '\t') value++;
        int len = strlen(value);
        while (len > 0 && (value[len - 1] == ' ' || value[len - 1] == '\t' || value[len - 1] == '\n' || value[len - 1] == '\r')) {
            value[--len] = '\0';
        }

        if (strcmp(key, "dataset") == 0) {
            dataset = value;
            if (!*dataset) {
                fprintf(stderr, "Falha ao alocar memória para dataset\n");
                fclose(fp);
                return NULL;
            }
        }
        else if (strcmp(key, "train_size") == 0) *train_size = atoi(value);
        else if (strcmp(key, "test_size") == 0) *test_size = atoi(value);
        else if (strcmp(key, "columns") == 0) *columns = atoi(value);
        else if (strcmp(key, "learning_rate") == 0) {
            config -> learning_rate = atof(value);
            config -> current_learning_rate = atof(value);
        }
        else if (strcmp(key, "batch_size") == 0) config -> batch_size = atoi(value);
        else if (strcmp(key, "decay_rate") == 0) config -> decay_rate = atof(value);
        else if (strcmp(key, "epsilon") == 0) config -> epsilon = atof(value);
        else if (strcmp(key, "epochs") == 0) config -> epochs = atoi(value);
        else if (strcmp(key, "beta_1") == 0) config -> beta_1 = atof(value);
        else if (strcmp(key, "beta_2") == 0) config -> beta_2 = atof(value);
        else printf("Chave desconhecida ignorada: %s\n", key);

    }
    fclose(fp);

    return dataset;
}

void get_data_from_csv(char *csv, float **X, float **y, int rows, int columns) {
    FILE *file;
    char row[1024];

    file = fopen(csv, "r");
    if (file == NULL) {
        printf("Error when opening the file.\n");
        return;
    }

    char *token = strtok(row, ",\n");

    *X = (float *) malloc((columns - 1) * rows * sizeof(float));
    *y = (float *) malloc(rows * sizeof(float));

    int i = 0;
    while (fgets(row, sizeof(row), file) && i < rows) {
        token = strtok(row, ",\n");
        int j = 0;
        while (token != NULL && j < columns) {
            float value = atof(token);
            if (j < columns - 1) (*X)[i * (columns - 1) + j] = value;
            else (*y)[i] = value;
            token = strtok(NULL, ",\n");
            j++;
        }

        i++;
    }

    fclose(file);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void load_weights(int id, float *weights, int rows, int columns) {
    FILE *file;
    char filename[100];
    snprintf(filename, sizeof(filename), "./datasets/sine/weights%i.csv", id + 1);

    file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error when opening the file: %s\n", filename);
        return;
    }

    char line[columns * 32];  // linha suficientemente grande
    int i = 0;

    while (fgets(line, sizeof(line), file) && i < rows) {
        line[strcspn(line, "\r\n")] = 0;  // remove \n ou \r

        int j = 0;
        char *token = strtok(line, ",");
        while (token && j < columns) {
            weights[i * columns + j] = atof(token);
            token = strtok(NULL, ",");
            j++;
        }

        // Se a linha tiver menos colunas que o esperado, preenche com 0
        while (j < columns) {
            weights[i * columns + j] = 0.0f;
            j++;
        }

        i++;
    }

    // Se faltarem linhas no arquivo, preenche com zeros
    while (i < rows) {
        for (int j = 0; j < columns; j++) {
            weights[i * columns + j] = 0.0f;
        }
        i++;
    }

    fclose(file);
}



#endif

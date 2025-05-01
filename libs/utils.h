
#ifndef NN_UTILS_H
#define NN_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "structs.h"

float *float_vector(int size) {
    return (float *) malloc(size * sizeof(float));
}

void show_matrix(Matrix m, int limit) {
    if (limit == 0) return;

    int end = (limit > 0 && limit <= m.dims[0]) ? limit : m.dims[0];

    for (int i = 0; i < end; i++) {
        printf("[");
        for (int j = 0; j < m.dims[1]; j++) {
            printf("%.10f, ", m.data[i * m.dims[1] + j]);
        }
        printf("],\n");
    }
    printf("\nSize: %i x %i\n============\n", m.dims[0], m.dims[1]);
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

void write_metrics(int epoch, float acc, float loss, char *dataset) {
    static const char *base_dir = "./datasets/";
    char *file_path = (char *) malloc(512 * sizeof(char));
    sprintf(file_path, "%s%s/metrics.csv", base_dir, dataset);

    FILE *arquivo = fopen(file_path, "a");

    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo!\n");
        return;
    }

    fprintf(arquivo, "%i,%f,%f\n", epoch, acc, loss);
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
        else if (strcmp(key, "learning_rate") == 0) config->learning_rate = atof(value);
        else if (strcmp(key, "batch_size") == 0) config->batch_size = atoi(value);
        else if (strcmp(key, "decay_rate") == 0) config->decay_rate = atof(value);
        else if (strcmp(key, "epsilon") == 0) config->epsilon = atof(value);
        else if (strcmp(key, "epochs") == 0) config->epochs = atoi(value);
        else if (strcmp(key, "beta_1") == 0) config->beta_1 = atof(value);
        else if (strcmp(key, "beta_2") == 0) config->beta_2 = atof(value);
        else printf("Chave desconhecida ignorada: %s\n", key);

    }
    fclose(fp);

    return dataset;
}

void get_data_from_csv(char *csv, Matrix *X, Matrix *y, int rows, int columns) {
    FILE *file;
    char row[1024];

    file = fopen(csv, "r");
    if (file == NULL) {
        printf("Error when opening the file.\n");
        return;
    }

    char *token = strtok(row, ",\n");

    float *X_, *y_;
    X_ = (float *) malloc((columns - 1) * rows * sizeof(float));
    y_ = (float *) malloc(rows * sizeof(float));

    int i = 0;
    while (fgets(row, sizeof(row), file) && i < rows) {
        token = strtok(row, ",\n");
        int j = 0;
        while (token != NULL && j < columns) {
            float value = atof(token);
            if (j < columns - 1) X_[i * (columns - 1) + j] = value;
            else y_[i] = value;
            token = strtok(NULL, ",\n");
            j++;
        }

        i++;
    }

    *X = init_matrix(X_, rows, columns - 1);
    *y = init_matrix(y_, rows, 1);

    fclose(file);
}

#endif

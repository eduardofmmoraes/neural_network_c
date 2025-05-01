
#ifndef NN_STRUCTS_H
#define NN_STRUCTS_H

#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979323846

typedef struct {
    float *data;
    int dims[2];
} Matrix;

typedef struct {
    Matrix weights;
    Matrix biases;
    const char *activation;
    Matrix inputs;
    Matrix z;
    Matrix outputs;

    Matrix weight_momentums;
    Matrix bias_momentums;

    Matrix weight_cache;
    Matrix bias_cache;

    float weight_regularizer_l1;
    float weight_regularizer_l2;
    float bias_regularizer_l1;
    float bias_regularizer_l2;
} Layer;

typedef struct {
    int batch_size;
    int epochs;
    float learning_rate;
    float current_learning_rate;
    float decay_rate;
    float epsilon;
    float beta_1;
    float beta_2;
    int iterations;
} Config;

Matrix init_matrix(float *data, int rows, int columns) {
    Matrix m;
    m.data = data;
    m.dims[0] = rows;
    m.dims[1] = columns;

    return m;
}

float random_weight() {
    float u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    float u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

Layer init_layer(int input_size, int size, const char *activation,
        float weight_regularizer_l1,
        float weight_regularizer_l2,
        float bias_regularizer_l1,
        float bias_regularizer_l2) {
    Layer layer;

    float *weights = (float *) malloc(input_size * size * sizeof(float));
    for (int i = 0; i < input_size * size; i++) {
        weights[i] = random_weight();
    }

    layer.weight_regularizer_l1 = weight_regularizer_l1;
    layer.weight_regularizer_l2 = weight_regularizer_l2;
    layer.bias_regularizer_l1 = bias_regularizer_l1;
    layer.bias_regularizer_l2 = bias_regularizer_l2;

    float *biases = (float *) calloc(size, sizeof(float));

    float *weight_mom = (float *) calloc(input_size * size, sizeof(float));
    float *weight_cac = (float *) calloc(input_size * size, sizeof(float));

    float *bias_mom = (float *) calloc(size, sizeof(float));
    float *bias_cac = (float *) calloc(size, sizeof(float));

    layer.weights = init_matrix(weights, input_size, size);
    layer.weight_momentums = init_matrix(weight_mom, input_size, size);
    layer.weight_cache = init_matrix(weight_cac, input_size, size);

    layer.biases = init_matrix(biases, size, 1);
    layer.bias_momentums = init_matrix(bias_mom, size, 1);
    layer.bias_cache = init_matrix(bias_cac, size, 1);

    layer.activation = activation;

    return layer;
}

#endif

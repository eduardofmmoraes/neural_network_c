
#ifndef NN_STRUCTS_H
#define NN_STRUCTS_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PI 3.14159265358979323846

typedef struct {
    int size;
    int input_size;

    float *weights;
    float *biases;

    float *inputs;
    float *outputs;

    char *activation;
    float *activation_outputs;

    float *dX;
    float *dW;
    float *dB;

    float *weight_momentums;
    float *bias_momentums;

    float *weight_cache;
    float *bias_cache;

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

float rand_uniform() {
    return 2.0f * ((float) rand() / RAND_MAX) - 1.0f;
}

Layer *init_layer(int n_inputs, int size, char *activation,
        float weight_regularizer_l1,
        float weight_regularizer_l2,
        float bias_regularizer_l1,
        float bias_regularizer_l2,
        int batch_size) {
    Layer *layer = (Layer *) malloc(sizeof(Layer));

    layer -> size = size;
    layer -> input_size = n_inputs;

    float limit = sqrtf(6.0f / (n_inputs + size));

    int weight_size = n_inputs * size;
    int bias_size = size;
    int output_size = batch_size * size;
    int input_size = batch_size * n_inputs;

    layer -> inputs = (float *) malloc(input_size * sizeof(float));
    layer -> dX = (float *) malloc(input_size * sizeof(float));

    layer -> weights = (float *) malloc(weight_size * sizeof(float));
    for (int i = 0; i < weight_size; i++) layer -> weights[i] = 0.1 * rand_uniform() * limit;
    layer -> weight_regularizer_l1 = weight_regularizer_l1;
    layer -> weight_regularizer_l2 = weight_regularizer_l2;
    layer -> weight_momentums = (float *) calloc(weight_size, sizeof(float));
    layer -> weight_cache = (float *) calloc(weight_size, sizeof(float));
    layer -> dW = (float *) malloc(weight_size * sizeof(float));

    layer -> biases = (float *) calloc(size, sizeof(float));
    layer -> bias_regularizer_l1 = bias_regularizer_l1;
    layer -> bias_regularizer_l2 = bias_regularizer_l2;
    layer -> bias_momentums = (float *) calloc(size, sizeof(float));
    layer -> bias_cache = (float *) calloc(size, sizeof(float));
    layer -> dB = (float *) malloc(bias_size * sizeof(float));

    layer -> outputs = (float *) malloc(output_size * sizeof(float));

    layer -> activation = activation;
    layer -> activation_outputs = (float *) malloc(output_size * sizeof(float));

    return layer;
}

void free_layer(Layer *layer) {
    free(layer -> inputs);
    free(layer -> dX);
    free(layer -> weights);
    free(layer -> weight_momentums);
    free(layer -> weight_cache);
    free(layer -> dW);
    free(layer -> biases);
    free(layer -> bias_momentums);
    free(layer -> bias_cache);
    free(layer -> dB);
    free(layer -> outputs);
    free(layer);
}

#endif

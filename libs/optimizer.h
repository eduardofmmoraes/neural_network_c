
#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structs.h"

void pre_update(Config *config) {
    config -> current_learning_rate = config -> learning_rate / (1 + config -> decay_rate * config -> iterations);
}

void update_params_SGD(Layer *layer, Matrix dW, Matrix dB, float lr, float momentum) {
    
    for (int j = 0; j < layer -> weights.dims[1]; j++) {
        for (int i = 0; i < layer -> weights.dims[0]; i++) {
            int idx = i * layer -> weights.dims[1] + j;
            float weight_update = momentum * layer -> weight_momentums.data[idx] - lr * dW.data[idx];
            layer -> weight_momentums.data[idx] = weight_update;
            layer -> weights.data[idx] += weight_update;
        }
        float bias_update = momentum * layer -> bias_momentums.data[j] - lr * dB.data[j];
        layer -> bias_momentums.data[j] = bias_update;
        layer -> biases.data[j] += bias_update;
    }
}

void update_params_Adam(Layer *layer, Matrix dW, Matrix dB, Config config) {

    float lr = config.current_learning_rate;
    float beta_1 = config.beta_1;
    float beta_2 = config.beta_2;
    float epsilon = config.epsilon;
    int iterations = config.iterations;
    
    for (int j = 0; j < layer -> weights.dims[1]; j++) {
        for (int i = 0; i < layer -> weights.dims[0]; i++) {
            int idx = i * layer -> weights.dims[1] + j;

            layer -> weight_momentums.data[idx] = beta_1 * layer -> weight_momentums.data[idx] + (1 - beta_1) * dW.data[idx];
            float weight_momentum_corrected = layer -> weight_momentums.data[idx] / (1 - pow(beta_1, iterations + 1));

            layer -> weight_cache.data[idx] = beta_2 * layer -> weight_cache.data[idx] + (1 - beta_2) * dW.data[idx] * dW.data[idx];
            float weight_cache_corrected = layer -> weight_cache.data[idx] / (1 - pow(beta_2, iterations + 1));

            layer -> weights.data[idx] += - lr * weight_momentum_corrected / (sqrt(weight_cache_corrected) + epsilon);
        }

        layer -> bias_momentums.data[j] = beta_1 * layer -> bias_momentums.data[j] + (1 - beta_1) * dB.data[j];
        float bias_momentum_corrected = layer -> bias_momentums.data[j] / (1 - pow(beta_1, iterations + 1));

        layer -> bias_cache.data[j] = beta_2 * layer -> bias_cache.data[j] + (1 - beta_2) * dB.data[j] * dB.data[j];
        float bias_cache_corrected = layer -> bias_cache.data[j] / (1 - pow(beta_2, iterations + 1));

        layer -> biases.data[j] += - lr * bias_momentum_corrected / (sqrt(bias_cache_corrected) + epsilon);
    }
}

#endif

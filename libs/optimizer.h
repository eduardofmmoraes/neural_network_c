
#ifndef NN_OPTIMIZER_H
#define NN_OPTIMIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structs.h"
#include "utils.h"

void pre_update(Config *config) {
    config -> current_learning_rate = config -> learning_rate / (1 + config -> decay_rate * config -> iterations);
}

void pos_update(Config *config) {
    (config -> iterations)++;
}

void update_params_Adam(Layer *layer, Config config) {
    int iterations = config.iterations;
    float lr = config.current_learning_rate;
    float beta_1 = config.beta_1;
    float beta_1_t = powf(beta_1, iterations + 1);
    float beta_2 = config.beta_2;
    float beta_2_t = powf(beta_2, iterations + 1);
    float epsilon = config.epsilon;
    
    for (int j = 0; j < layer -> size; j++) {
        for (int i = 0; i < layer -> input_size; i++) {
            int idx = i * layer -> size + j;

            layer -> weight_momentums[idx] = beta_1 * layer -> weight_momentums[idx] + (1 - beta_1) * layer -> dW[idx];
            float weight_momentum_corrected = layer -> weight_momentums[idx] / (1 - beta_1_t);

            layer -> weight_cache[idx] = beta_2 * layer -> weight_cache[idx] + (1 - beta_2) * layer -> dW[idx] * layer -> dW[idx];
            float weight_cache_corrected = layer -> weight_cache[idx] / (1 - beta_2_t);

            layer -> weights[idx] +=  - lr * weight_momentum_corrected / (sqrtf(weight_cache_corrected) + epsilon);
        }

        layer -> bias_momentums[j] = beta_1 * layer -> bias_momentums[j] + (1 - beta_1) * layer -> dB[j];
        float bias_momentum_corrected = layer -> bias_momentums[j] / (1 - beta_1_t);

        layer -> bias_cache[j] = beta_2 * layer -> bias_cache[j] + (1 - beta_2) * layer -> dB[j] * layer -> dB[j];
        float bias_cache_corrected = layer -> bias_cache[j] / (1 - beta_2_t);

        layer -> biases[j] += - lr * bias_momentum_corrected / (sqrt(bias_cache_corrected) + epsilon);
    }
}

#endif

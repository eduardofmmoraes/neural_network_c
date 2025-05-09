
#ifndef NN_TRAIN_H
#define NN_TRAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "structs.h"
#include "activation_functions.h"

void choose_activation_function(Layer *layer, int n) {
    if (strcmp(layer -> activation, "relu") == 0) ReLU(layer, n);
    else if (strcmp(layer -> activation, "tanh") == 0) Tanh(layer, n);
    else return Linear(layer, n);
}

void forward(float *inputs, Layer *layer, int n) {
    layer -> inputs = inputs;

    int i_dims[2] = {n, layer -> input_size};
    int w_dims[2] = {layer -> input_size, layer -> size};
    
    for (int i = 0; i < i_dims[0]; i++) {
        for (int j = 0; j < w_dims[1]; j++) {
            float sum = 0;
            for (int k = 0; k < w_dims[0]; k++) 
                sum += inputs[i * i_dims[1] + k] * layer -> weights[k * w_dims[1] + j];
            layer -> outputs[i * w_dims[1] + j] = sum + layer -> biases[j];
        }
    }

    choose_activation_function(layer, n);
}

void calculate_dX(float *grad, Layer *layer, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < layer -> input_size; j++) {
            float sum = 0;
            for (int k = 0; k < layer -> size; k++) {
                sum += grad[i * layer -> size + k] * layer -> weights[j * layer -> size + k];
            }
            layer -> dX[i * layer -> input_size + j] = sum;
        }
    }
}

void calculate_dW(float *grad, Layer *layer, int n) {
    float dL1;
    
    for (int i = 0; i < layer -> input_size; i++) {
        for (int j = 0; j < layer -> size; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += layer -> inputs[k * layer -> input_size + i] * grad[k * layer -> size + j];
            }

            if (layer -> weight_regularizer_l1 > 0) {
                dL1 = layer -> weights[i * layer -> size + j] < 0 ? -1 : 1;
                sum += layer -> weight_regularizer_l1 * dL1;
            }
            
            if (layer -> weight_regularizer_l2 > 0) {
                sum += 2 * layer -> weight_regularizer_l2 * layer -> weights[i * layer -> size + j];
            }

            layer -> dW[i * layer -> size + j] = sum;
        }
    }
}

void calculate_dB(float *grad, Layer *layer, int n) {
    float dL1;

    for (int i = 0; i < layer -> size; i++) {
        float sum = 0;
        for (int j = 0; j < n; j++) sum += grad[j * layer -> size + i];
        
        if (layer -> bias_regularizer_l1 > 0) {
            dL1 = layer -> biases[i] < 0 ? -1 : 1;
            sum += layer -> bias_regularizer_l1 * dL1;
        }
        
        if (layer -> bias_regularizer_l2 > 0) {
            sum += 2 * layer -> bias_regularizer_l2 * layer -> biases[i];
        }

        layer -> dB[i] = sum;
    }
}

void backward(float *grad, Layer *layer, int n) {
    if (strcmp(layer -> activation, "relu") == 0) dReLU(grad, layer, n);
    else if (strcmp(layer -> activation, "tanh") == 0) dTanh(grad, layer, n);
    else dLinear(grad, layer, n);

    calculate_dW(grad, layer, n);
    calculate_dB(grad, layer, n);
    calculate_dX(grad, layer, n);
}

#endif

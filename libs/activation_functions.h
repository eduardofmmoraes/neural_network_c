#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structs.h"
#include "utils.h"

void Linear(Layer *layer, int n) {
    memcpy(layer -> activation_outputs, layer -> outputs, n * layer -> size * sizeof(float));
}

void dLinear(float *grad, Layer *layer, int n) {}

void ReLU(Layer *layer, int n) {
    float value = 0;
    for (int i = 0; i < n * layer -> size; i++) {
        value = layer -> outputs[i];
        layer -> activation_outputs[i] = (value > 0) ? value : 0;
    }
}

void dReLU(float *grad, Layer *layer, int n) {
    float value;
    for (int i = 0; i < n * layer -> size; i++) {
        value = layer -> outputs[i];
        grad[i] = value > 0.0 ? grad[i] : 0.0;
    }
}

void Tanh(Layer *layer, int n) {
    for (int i = 0; i < n * layer -> size; i++) {
        layer -> activation_outputs[i] = tanh(layer -> outputs[i]);
    }
}

void dTanh(float *grad, Layer *layer, int n) {
    float tanh_v;
    for (int i = 0; i < n * layer -> size; i++) {
        tanh_v = layer -> activation_outputs[i];
        grad[i] = grad[i] * (1 - tanh_v * tanh_v);
    }
}

#endif


#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structs.h"
#include "utils.h"

Matrix Linear(Matrix z, Layer *layer) {
    layer -> z = z;
    layer -> outputs = z;
    return layer -> outputs;
}

Matrix ReLU(Matrix z, Layer *layer) {
    float *data = float_vector(z.dims[0] * z.dims[1]);

    
    for (int i = 0; i < z.dims[0] * z.dims[1]; i++) {
        data[i] = (z.data[i] > 0) ? z.data[i] : 0;
    }

    layer -> z = z;
    layer -> outputs = init_matrix(data, z.dims[0], z.dims[1]);
    return layer -> outputs;
}

Matrix Leaky_ReLU(Matrix z, Layer *layer) {
    float *data = float_vector(z.dims[0] * z.dims[1]);

    
    for (int i = 0; i < z.dims[0] * z.dims[1]; i++) {
        data[i] = z.data[i] * ((z.data[i] > 0) ? 1 : 0.01);
    }

    layer -> z = z;
    layer -> outputs = init_matrix(data, z.dims[0], z.dims[1]);
    return layer -> outputs;
}

Matrix Sigmoid(Matrix z, Layer *layer) {
    float *data = float_vector(z.dims[0] * z.dims[1]);

    
    for (int i = 0; i < z.dims[0] * z.dims[1]; i++) {
        data[i] = 1 / (1 + exp(-z.data[i]));
    }

    layer -> z = z;
    layer -> outputs = init_matrix(data, z.dims[0], z.dims[1]);
    return layer -> outputs;
}

Matrix Tanh(Matrix z, Layer *layer) {
    float *data = float_vector(z.dims[0] * z.dims[1]);

    
    for (int i = 0; i < z.dims[0] * z.dims[1]; i++) {
        data[i] = tanh(z.data[i]);
    }

    layer -> z = z;
    layer -> outputs = init_matrix(data, z.dims[0], z.dims[1]);
    return layer -> outputs;
}

Matrix dLinear(Matrix grad, Layer layer) {
    float *data = float_vector(layer.outputs.dims[0] * layer.outputs.dims[1]);

    return init_matrix(data, layer.outputs.dims[0], layer.outputs.dims[1]);
}

Matrix dReLU(Matrix grad, Layer layer) {
    float *data = float_vector(layer.outputs.dims[0] * layer.outputs.dims[1]);

    
    for (int i = 0; i < layer.outputs.dims[0] * layer.outputs.dims[1]; i++) {
        data[i] = layer.z.data[i] > 0 ? grad.data[i] : 0;
    }

    return init_matrix(data, layer.outputs.dims[0], layer.outputs.dims[1]);
}

Matrix dLeaky_ReLU(Matrix grad, Layer layer) {
    float *data = float_vector(layer.outputs.dims[0] * layer.outputs.dims[1]);

    
    for (int i = 0; i < layer.outputs.dims[0] * layer.outputs.dims[1]; i++) {
        data[i] = grad.data[i] * (layer.z.data[i] > 0 ? 1 : 0.01);
    }

    return init_matrix(data, layer.outputs.dims[0], layer.outputs.dims[1]);
}

Matrix dSigmoid(Matrix grad, Layer layer) {
    float *data = float_vector(layer.outputs.dims[0] * layer.outputs.dims[1]);

    
    for (int i = 0; i < layer.outputs.dims[0] * layer.outputs.dims[1]; i++) {
        data[i] = grad.data[i] * layer.outputs.data[i] * (1 - layer.outputs.data[i]);
    }

    return init_matrix(data, layer.outputs.dims[0], layer.outputs.dims[1]);
}

Matrix dTanh(Matrix grad, Layer layer) {
    float *data = float_vector(layer.outputs.dims[0] * layer.outputs.dims[1]);

    for (int i = 0; i < layer.outputs.dims[0] * layer.outputs.dims[1]; i++) {
        data[i] = grad.data[i] * (1 - layer.outputs.data[i] * layer.outputs.data[i]);
    }

    return init_matrix(data, layer.outputs.dims[0], layer.outputs.dims[1]);
}

#endif

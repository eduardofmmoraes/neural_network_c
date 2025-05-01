
#ifndef NN_TRAIN_H
#define NN_TRAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "structs.h"
#include "activation_functions.h"

Matrix *to_batches(Matrix X, int batch_size, int n) {
    int start = 0,
        end = 0;
    int n_columns = X.dims[1];
    Matrix *batches = (Matrix *) malloc(n * sizeof(Matrix));
    for (int i = 0; i < n; i++) {
        start = i * batch_size;
        end = (i < n - 1) ? (i + 1) * batch_size : X.dims[0];
        float *data = (float *) malloc((end - start) * n_columns * sizeof(float));
        for (int j = start; j < end; j++) {
            for (int k = 0; k < n_columns; k++) {
                data[(j - start) * n_columns + k] = X.data[j * n_columns + k];
            }
        }
        batches[i] = init_matrix(data, end - start, n_columns);
    }

    return batches;
}

Matrix choose_activation_function(Matrix z, Layer *layer) {
    if (strcmp(layer -> activation, "relu") == 0) return ReLU(z, layer);
    else if (strcmp(layer -> activation, "sigmoid") == 0) return Sigmoid(z, layer);
    else if (strcmp(layer -> activation, "tanh") == 0) return Tanh(z, layer);
    else if (strcmp(layer -> activation, "leaky_relu") == 0) return Leaky_ReLU(z, layer);
    else return Linear(z, layer);
}

Matrix forward(Matrix inputs, Layer *layer) {
    layer -> inputs = inputs;
    float *data = (float *) malloc(inputs.dims[0] * layer -> weights.dims[1] * sizeof(float));

    int layer_dims[2];
    layer_dims[0] = layer -> weights.dims[0];
    layer_dims[1] = layer -> weights.dims[1];

    int inputs_dims[2];
    inputs_dims[0] = inputs.dims[0];
    inputs_dims[1] = inputs.dims[1];

    
    for (int i = 0; i < inputs_dims[0]; i++) {
        for (int j = 0; j < layer_dims[1]; j++) {
            data[i * layer_dims[1] + j] = layer -> biases.data[j];
            for (int k = 0; k < inputs.dims[1]; k++) {
                data[i * layer_dims[1] + j] += inputs.data[i * inputs_dims[1] + k] * layer -> weights.data[k * layer_dims[1] + j];
            }
        }
    }

    Matrix z = init_matrix(data, inputs.dims[0], layer -> weights.dims[1]);
    return choose_activation_function(z, layer);
}

Matrix calculate_dX(Matrix grad, Layer layer) {
    float *dx = float_vector(grad.dims[0] * layer.weights.dims[0]);

    
    for (int i = 0; i < grad.dims[0]; i++) {
        for (int j = 0; j < layer.weights.dims[0]; j++) {
            dx[i * layer.weights.dims[0] + j] = 0;
            for (int k = 0; k < grad.dims[1]; k++) {
                dx[i * layer.weights.dims[0] + j] += grad.data[i * grad.dims[1] + k] * layer.weights.data[j * layer.weights.dims[1] + k];
            }
        }
    }

    return init_matrix(dx, grad.dims[0], layer.weights.dims[0]);
}

Matrix calculate_dW(Layer layer, Matrix grad) {
    float *dw = float_vector(layer.inputs.dims[1] * grad.dims[1]);
    float dL1;

    
    for (int i = 0; i < layer.inputs.dims[1]; i++) {
        for (int j = 0; j < grad.dims[1]; j++) {
            int idx = i * layer.weights.dims[1] + j;
            dw[idx] = 0;
            for (int k = 0; k < grad.dims[0]; k++) {
                dw[idx] += layer.inputs.data[k * layer.inputs.dims[1] + i] * grad.data[k * grad.dims[1] + j];
            }

            if (layer.weight_regularizer_l1 > 0) {
                dL1 = layer.weights.data[idx] < 0 ? -1 : 1;
                dw[idx] += layer.weight_regularizer_l1 * dL1;
            }

            if (layer.weight_regularizer_l2 > 0) {
                dw[idx] += 2 * layer.weight_regularizer_l2 * layer.weights.data[idx];
            }
        }
    }

    return init_matrix(dw, layer.inputs.dims[1], grad.dims[1]);
}

Matrix calculate_dB(Layer layer, Matrix grad) {
    float *db = float_vector(grad.dims[1]);
    float dL1;

    
    for (int i = 0; i < grad.dims[1]; i++) {
        db[i] = 0;
        for (int j = 0; j < grad.dims[0]; j++) {
            db[i] += grad.data[j * grad.dims[1] + i];
        }

        if (layer.bias_regularizer_l1 > 0) {
            dL1 = layer.biases.data[i] < 0 ? -1 : 1;
            db[i] += layer.bias_regularizer_l1 * dL1;
        }

        if (layer.bias_regularizer_l2 > 0) {
            db[i] += 2 * layer.bias_regularizer_l2 * layer.biases.data[i];
        }
    }

    return init_matrix(db, grad.dims[1], 1);
}

Matrix backward(Matrix grad, Layer *layer, Matrix *dW, Matrix *dB) {
    if (strcmp(layer -> activation, "relu") == 0) grad = dReLU(grad, *layer);
    else if (strcmp(layer -> activation, "sigmoid") == 0) grad = dSigmoid(grad, *layer);
    else if (strcmp(layer -> activation, "tanh") == 0) grad = dTanh(grad, *layer);
    else if (strcmp(layer -> activation, "leaky relu") == 0) grad = dLeaky_ReLU(grad, *layer);
    else dLinear(grad, *layer);

    *dW = calculate_dW(*layer, grad);
    *dB = calculate_dB(*layer, grad);

    return calculate_dX(grad, *layer);
}

#endif

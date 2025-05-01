#ifndef NN_METRICS_H
#define NN_METRICS_H

#include <stdio.h>
#include <stdlib.h>

#include "structs.h"
#include "utils.h"

Matrix Error(Matrix y, Matrix output) {
    int n = output.dims[0];
    float *data = float_vector(n);

    for (int i = 0; i < n; i++) {
        data[i] = y.data[i] - output.data[i];
    }

    return init_matrix(data, n, 1);
}

void Metrics_and_Gradients(Matrix error, float *loss, float *rmse, Matrix *mse_grad) {
    float total_acc = 0;
    float total_mse = 0;

    int n = error.dims[0];
    float *grad = float_vector(n);

    for (int i = 0; i < n; i++) {
        total_mse += error.data[i] * error.data[i];
        grad[i] = -(2.0 / n) * error.data[i];
    }

    *loss = total_mse / n;
    *rmse = sqrtf(*loss);
    *mse_grad = init_matrix(grad, n, 1);
}

#endif
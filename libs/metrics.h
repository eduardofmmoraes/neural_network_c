#ifndef NN_METRICS_H
#define NN_METRICS_H

#include <stdio.h>
#include <stdlib.h>

#include "structs.h"
#include "utils.h"

void Error(float *y, float *output, float *error, int n) {
    for (int i = 0; i < n; i++) error[i] = y[i] - output[i];
}

void Metrics(float *error, float *loss, float *rmse, float *acc, float precision, int n) {
    float total_acc = 0;
    float total_mse = 0;

    for (int i = 0; i < n; i++) {
        total_mse += error[i] * error[i];
        total_acc += fabsf(error[i]) < precision;
    }

    *loss = total_mse / n;
    *rmse = sqrtf(*loss);
    *acc = total_acc / n;
}

void Gradient(float *error, float *grad, int n) {
    for (int i = 0; i < n; i++) grad[i] = -2.0 * error[i] / n;
}

#endif
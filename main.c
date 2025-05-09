#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "./libs/structs.h"
#include "./libs/optimizer.h"
#include "./libs/train.h"
#include "./libs/utils.h"
#include "./libs/metrics.h"

#define NUM_LAYERS 3

float std(float *m, int n) {
    float mean, sum = 0, std, sum_diff = 0, value;
    for (int i = 0; i < n; i++) sum += m[i];
    mean = sum / n;

    for (int i = 0; i < n; i++) {
        value = m[i] - mean;
        sum_diff += (value * value);
    }

    return sqrtf(sum_diff / (n - 1));
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    char *dataset = argv[1];
    int train_rows, test_rows, columns;

    Config config;
    load_config("./config.txt", &config, &train_rows, &test_rows, &columns);
        
    float *X_train, *y_train, *X_test, *y_test;
    char **paths = get_paths(dataset);

    get_data_from_csv(paths[0], &X_train, &y_train, train_rows, columns);
    get_data_from_csv(paths[1], &X_test, &y_test, test_rows, columns);

    free(paths[0]);
    free(paths[1]);
    free(paths);
    
    Layer *layers[NUM_LAYERS] = {
        init_layer(columns - 1, 64, "relu", 0, 0, 0, 0, train_rows),
        init_layer(64, 64, "relu", 0, 0, 0, 0, train_rows),
        init_layer(64, 1, "linear", 0, 0, 0, 0, train_rows)
    };

    float *grad;
    float *error = (float *) malloc(train_rows * sizeof(float)),
            *dloss = (float *) malloc(train_rows * sizeof(float));

    float precision = 1e-6;

    int step = 0;
    for (int i = 0; i < config.epochs; i++) {
        float loss = 0, rmse = 0, acc = 0;
        printf("==== Epoch %i ====\n", i + 1);

        for (int k = 0; k < NUM_LAYERS; k++) {
            forward((k == 0) ? X_train : layers[k - 1] -> activation_outputs, layers[k], train_rows);
        }
        
        for (int k = 0; k < train_rows; k++) {
            error[k] = y_train[k] - layers[NUM_LAYERS - 1] -> activation_outputs[k];
            dloss[k] = -2.0 * error[k] / train_rows;
        }

        Metrics(error, &loss, &rmse, &acc, precision, train_rows);
        
        pre_update(&config);
        grad = dloss;
        for (int k = NUM_LAYERS - 1; k >= 0; k--) {
            backward((k == NUM_LAYERS - 1) ? dloss : layers[k + 1] -> dX, layers[k], train_rows);
            // if (k == 0) show_matrix(layers[k] -> dW, -1, layers[k] -> input_size, layers[k] -> size);
            // update_params_Adam(layers[k], config);

            for (int b = 0; b < layers[k] -> size; b++) {
                for (int a = 0; a < layers[k] -> input_size; a++) {
                    int idx = a * layers[k] -> size + b;
                    layers[k] -> weights[idx] = -config.current_learning_rate * layers[k] -> dW[idx];
                }

                layers[k] -> biases[b] = -config.current_learning_rate * layers[k] -> dB[b];
            }
        }
        pos_update(&config);

        printf("RMSE: %.5lf\n", rmse);
        printf("Loss (MSE): %.5lf\n", loss);
        printf("Accuracy: %.5lf\n", acc);
        write_metrics(i + 1, rmse, loss, acc, dataset);

        write_cmp(y_train, layers[NUM_LAYERS - 1] -> activation_outputs, train_rows, dataset);

    }

    float acc = 0, loss = 0, rmse = 0;
    printf("======== TEST ========\n");
    for (int i = 0; i < NUM_LAYERS; i++) {
        forward((i == 0) ? X_test : layers[i - 1] -> activation_outputs, layers[i], test_rows);
    }

    write_pred(layers[NUM_LAYERS - 1] -> activation_outputs, dataset, test_rows);
    Error(y_test, layers[NUM_LAYERS - 1] -> activation_outputs, error, test_rows);
    Metrics(error, &loss, &rmse, &acc, precision, test_rows);
    printf("RMSE: %.5lf\n", rmse);
    printf("Loss (MSE): %.5lf\n", loss);
    printf("Accuracy: %.5lf\n", acc);

    for (int i = 0; i < NUM_LAYERS; i++) free_layer(layers[i]);  

    return 0;
}
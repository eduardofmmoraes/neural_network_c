
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

#define NUM_LAYERS 4

int main(int argc, char *argv[]) {
    srand(time(NULL));

    char *dataset = argv[1];
    int train_rows, test_rows, columns;

    Config config;
    load_config("./config.txt", &config,
                &train_rows, &test_rows, &columns);

    Matrix X_train, y_train, X_test, y_test;
    char **paths = get_paths(dataset);

    printf("%s\n", dataset);

    get_data_from_csv(paths[0], &X_train, &y_train, train_rows, columns);
    get_data_from_csv(paths[1], &X_test, &y_test, test_rows, columns);

    free(paths[0]);
    free(paths[1]);
    free(paths);

    int n_batches;

    if (config.batch_size > 0) {
        n_batches = (int) (train_rows / config.batch_size);
        if (n_batches * config.batch_size < train_rows) n_batches += 1;
    } else {
        config.batch_size = train_rows;
        n_batches = 1;
    }

    Matrix *X_train_batches = to_batches(X_train, config.batch_size, n_batches);
    Matrix *y_train_batches = to_batches(y_train, config.batch_size, n_batches);

    Layer layers[NUM_LAYERS] = {
        init_layer(X_train.dims[1], 32, "relu", 0, 5e-4, 0, 5e-4),
        init_layer(32, 32, "relu", 0, 5e-4, 0, 5e-4),
        init_layer(32, 16, "relu", 0, 5e-4, 0, 5e-4),
        init_layer(16, 1, "linear", 0, 0, 0, 0),
    };

    Matrix output, grad, error, dX, dW, dB;

    int step = 0;
    for (int i = 0; i < config.epochs; i++) {
        float loss = 0, rmse = 0;
        printf("==== Epoch %i ====\n", i + 1);
        for (int j = 0; j < n_batches; j++) {
            float batch_loss = 0, batch_rmse = 0;
            output = X_train_batches[j];
            for (int k = 0; k < NUM_LAYERS; k++) {
                output = forward(output, &layers[k]);
            }

            error = Error(y_train_batches[j], output);
            Metrics_and_Gradients(error, &batch_loss, &batch_rmse, &grad);
            // printf("\t== Batch %i ==\n", j + 1);
            // printf("\tAccuracy: %.5lf\n", batch_acc);
            // printf("\tLoss: %.5lf\n", batch_loss);

            pre_update(&config);
            for (int k = NUM_LAYERS - 1; k >= 0; k--) {
                dX = backward(grad, &layers[k], &dW, &dB);
                update_params_Adam(&layers[k], dW, dB, config);
                grad = dX;
            }

            step++;
            loss += batch_loss;
            rmse += batch_rmse;
        }
        loss /= n_batches;
        rmse /= n_batches;
        printf("RMSE: %.5lf\n", rmse);
        printf("Loss: %.5lf\n", loss);
        write_metrics(i + 1, rmse, loss, dataset);
    }

    float acc = 0, loss = 0, rmse;
    printf("======== TEST ========\n");
    output = X_test;
    for (int i = 0; i < NUM_LAYERS; i++) {
        output = forward(output, &layers[i]);
    }
    error = Error(y_test, output);
    Metrics_and_Gradients(error, &loss, &rmse, &grad);
    printf("RMSE: %.5lf\n", rmse);
    printf("Loss: %.5lf\n", loss);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "./libs/structs.h"
// #include "./libs/optimizer.h"
// #include "./libs/train.h"
#include "./libs/utils.h"
// #include "./libs/metrics.h"

#define NUM_LAYERS 3

void write_matrix_to_file(const char *dataset, float *matrix, int rows, int cols, int id) {
    char *filename;
    sprintf(filename, "./datasets/%s/weights%i.csv", dataset, id);
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("Erro ao abrir o arquivo");
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(f, "%.10f", matrix[i * cols + j]);
            if (j < cols - 1)
                fprintf(f, ",");
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

void initialize_weights(float *weights, int dims[2], float mean, float sigma);
float std(float *m, int n);

int main(int argc, char *argv[]) {

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

    int w_dims[NUM_LAYERS][2] = {
        {columns - 1, 128},
        {128         , 128},
        {128         ,  1}
    };
    char *activations[NUM_LAYERS] = {
        "relu",
        "relu",
        "linear"
    };

    // creating the layers parameters
    float **inputs = (float **) malloc(NUM_LAYERS * sizeof(float*));
    float **outputs = (float **) malloc(NUM_LAYERS * sizeof(float *));
    float **activation_outputs = (float **) malloc(NUM_LAYERS * sizeof(float *));
    float **dX = (float **) malloc(NUM_LAYERS * sizeof(float *));
    float **grad = (float **) malloc(NUM_LAYERS * sizeof(float *));

    float **weights = (float **) malloc(NUM_LAYERS * sizeof(float *));
    float **dW = (float **) malloc(NUM_LAYERS * sizeof(float *));
    float **w_mom = (float **) calloc(NUM_LAYERS, sizeof(float *));
    float **w_cac = (float **) calloc(NUM_LAYERS, sizeof(float *));

    float **biases = (float **) calloc(NUM_LAYERS, sizeof(float *));
    float **dB = (float **) malloc(NUM_LAYERS * sizeof(float *));
    float **b_mom = (float **) calloc(NUM_LAYERS, sizeof(float *));
    float **b_cac = (float **) calloc(NUM_LAYERS, sizeof(float *));

    // Initializing paramters from each layer
    for (int i = 0; i < NUM_LAYERS; i++) {
        int i_size = train_rows * w_dims[i][0];
        inputs[i] = (float *) malloc(i_size * sizeof(float));
        dX[i] = (float *) malloc(i_size * sizeof(float));

        int o_size = train_rows * w_dims[i][1];
        outputs[i] = (float *) malloc(o_size * sizeof(float));
        activation_outputs[i] = (float *) malloc(o_size * sizeof(float));
        grad[i] = (float *) malloc(o_size * sizeof(float));

        int w_size = w_dims[i][0] * w_dims[i][1];
        weights[i] = (float *) malloc(w_size * sizeof(float));
        dW[i] = (float *) malloc(w_size * sizeof(float));
        w_mom[i] = (float *) calloc(w_size, sizeof(float));
        w_cac[i] = (float *) calloc(w_size, sizeof(float));
        
        load_weights(i, weights[i], w_dims[i][0], w_dims[i][1]);

        biases[i] = (float *) calloc(w_dims[i][1], sizeof(float));
        dB[i] = (float *) malloc(w_dims[i][1] * sizeof(float));
        b_mom[i] = (float *) calloc(w_dims[i][1], sizeof(float));
        b_cac[i] = (float *) calloc(w_dims[i][1], sizeof(float));
    }

    float error = 0;
    float *dloss = (float *) malloc(train_rows * sizeof(float));
    float *output;

    float lr = 0.001;
    float curr_lr = lr;
    float decay_rate = 0.001;
    float epsilon = 1e-7;
    float beta_1 = 0.9;
    float beta_2 = 0.999;
    int iterations = 0;
    float precision = std(y_train, train_rows) / 250;

    for (int i = 0; i < config.epochs; i++) {
        printf("Epoch %i: ", i + 1);

        // Forward pass
        for (int j = 0; j < NUM_LAYERS; j++) {
            memcpy(inputs[j], (j == 0) ? X_train : activation_outputs[j - 1], train_rows * w_dims[j][0] * sizeof(float));
            
            for (int k = 0; k < train_rows; k++) {
                for (int l = 0; l < w_dims[j][1]; l++) {
                    float sum = 0;
                    for (int m = 0; m < w_dims[j][0]; m++) {
                        sum += inputs[j][k * w_dims[j][0] + m] *
                                weights[j][m * w_dims[j][1] + l];
                    }

                    int output_idx = k * w_dims[j][1] + l;
                    outputs[j][output_idx] = sum + biases[j][l];

                    float act_value = outputs[j][output_idx];
                    
                    if (strcmp(activations[j], "relu") == 0) {
                        act_value = (act_value > 0) ? act_value : 0;
                    } else if (strcmp(activations[j], "tanh") == 0) {
                        act_value = tanhf(act_value);
                    }

                    activation_outputs[j][output_idx] = act_value;
                }
            }
        }
        
        // Gradient and metrics (MSE Loss, RMSE and precision)
        float loss = 0, rmse = 0, acc = 0, biggest_error = 0;
        biggest_error = -INFINITY;
        for (int j = 0; j < train_rows; j++) {
            error = y_train[j] - activation_outputs[NUM_LAYERS - 1][j];
            dloss[j] = -2.0 * error / train_rows;

            acc += fabsf(error) < precision;
            loss += error * error;

            if (error > biggest_error) biggest_error = error;
        }

        write_cmp(y_train, activation_outputs[NUM_LAYERS - 1], train_rows, dataset);

        acc /= train_rows;
        loss /= train_rows;
        rmse = sqrtf(loss);

        printf("RMSE: %.5f ", rmse);
        printf("Loss (MSE): %.5f ", loss);
        printf("Accuracy: %.5f ", acc);
        printf("Biggest error: %.5f ", biggest_error);
        printf("Learning rate: %.5f\n", curr_lr);
        
        // Backpropagation
        for (int j = NUM_LAYERS - 1; j >= 0; j--) {
            if (j == NUM_LAYERS - 1) {
                memcpy(grad[j], dloss, train_rows * sizeof(float));
            } else {
                memcpy(grad[j], dX[j + 1], train_rows * w_dims[j][1] * sizeof(float));
            }
            
            // Activation function gradient
            for (int k = 0; k < train_rows * w_dims[j][1]; k++) {
                float d_act = grad[j][k];
                if (strcmp(activations[j], "relu") == 0) {
                    d_act = (outputs[j][k] > 0) ? grad[j][k] : 0;
                } else if (strcmp(activations[j], "tanh") == 0) {
                    d_act = grad[j][k] * (1 - activation_outputs[j][k] * activation_outputs[j][k]);
                }
                
                grad[j][k] = d_act;
            }

            // Inputs gradient
            for (int k = 0; k < train_rows; k++) {
                for (int l = 0; l < w_dims[j][0]; l++) {
                    float sum = 0;
                    for (int m = 0; m < w_dims[j][1]; m++) {
                        sum += grad[j][k * w_dims[j][1] + m] * weights[j][l * w_dims[j][1] + m];
                    }
                    dX[j][k * w_dims[j][0] + l] = sum;
                }
            }

            // Weights gradient
            for (int k = 0; k < w_dims[j][0]; k++) {
                for (int l = 0; l < w_dims[j][1]; l++) {
                    float sum = 0;
                    for (int m = 0; m < train_rows; m++) {
                        sum += inputs[j][m * w_dims[j][0] + k] * grad[j][m * w_dims[j][1] + l];
                    }
                    dW[j][k * w_dims[j][1] + l] = sum;
                }
            }

            // Biases gradient
            for (int k = 0; k < w_dims[j][1]; k++) {
                float sum = 0;
                for (int l = 0; l < train_rows; l++) {
                    sum += grad[j][l * w_dims[j][1] + k];
                }
                dB[j][k] = sum;
            }
        }

        // Parameters update
        float beta_1_t = powf(beta_1, iterations + 1);
        float beta_2_t = powf(beta_2, iterations + 1);
        curr_lr = lr / (1.0 + decay_rate * iterations);
        for (int j = 0; j < NUM_LAYERS; j++) {
            // Adam optimizer
            for (int k = 0; k < w_dims[j][0]; k++) {
                for (int l = 0; l < w_dims[j][1]; l++) {
                    int idx = k * w_dims[j][1] + l;

                    w_mom[j][idx] = beta_1 * w_mom[j][idx] + (1 - beta_1) * dW[j][idx];
                    float w_mom_corrected = w_mom[j][idx] / (1 - beta_1_t);

                    w_cac[j][idx] = beta_2 * w_cac[j][idx] + (1 - beta_2) * dW[j][idx] * dW[j][idx];
                    float w_cac_corrected = w_cac[j][idx] / (1 - beta_2_t);

                    float w_update = -curr_lr * w_mom_corrected / (sqrtf(w_cac_corrected) + epsilon);

                    weights[j][idx] += w_update;
                
                    if (k == 0) {
                        b_mom[j][k] = beta_1 * b_mom[j][k] + (1 - beta_1) * dB[j][k];
                        float b_mom_corrected = b_mom[j][k] / (1 - beta_1_t);

                        b_cac[j][k] = beta_2 * b_cac[j][k] + (1 - beta_2) * dB[j][k] * dB[j][k];
                        float b_cac_corrected = b_cac[j][k] / (1 - beta_2_t);

                        float b_update = -curr_lr * b_mom_corrected / (sqrtf(b_cac_corrected) + epsilon);

                        biases[j][k] += b_update;
                    }
                }
            }
        }
        
        iterations++;
    }

    // float acc = 0, loss = 0, rmse = 0;
    // printf("======== TEST ========\n");
    // for (int i = 0; i < NUM_LAYERS; i++) {
    //     forward((i == 0) ? X_test : layers[i - 1] -> activation_outputs, layers[i], test_rows);
    // }

    // write_pred(layers[NUM_LAYERS - 1] -> activation_outputs, dataset, test_rows);
    // Error(y_test, layers[NUM_LAYERS - 1] -> activation_outputs, error, test_rows);
    // Metrics(error, &loss, &rmse, &acc, precision, test_rows);
    // printf("RMSE: %.5lf\n", rmse);
    // printf("Loss (MSE): %.5lf\n", loss);
    // printf("Accuracy: %.5lf\n", acc);
    
    return 0;
}

// void initialize_weights(float *weights, int dims[2], float mean, float sigma) {
//     const gsl_rng_type *T;
//     gsl_rng *r;

//     gsl_rng_env_setup();
//     T = gsl_rng_default;
//     r = gsl_rng_alloc(T);
//     gsl_rng_set(r, time(NULL));

//     for (int i = 0; i < dims[0] * dims[1]; i++) {
//         weights[i] = 0.1 * (mean + gsl_ran_gaussian(r, sigma));
//     }

//     gsl_rng_free(r);
// }

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
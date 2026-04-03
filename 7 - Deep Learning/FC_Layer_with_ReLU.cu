

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void fcForward(float* input, float* weights, float* bias,
                           float* output,
                           int batchSize, int inputSize, int outputSize) {

    // row = which image in the batch
    // col = which output neuron
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batchSize && col < outputSize) {

        // Start with bias for this output neuron
        float sum = bias[col];

        // Dot product : input[row] dot weights[:,col]
        // Loop over all 128 input features
        for (int k = 0; k < inputSize; k++) {
            sum += input[row * inputSize + k] *
                   weights[k * outputSize + col];
        }

        output[row * outputSize + col] = sum;
    }
}


// ReLU kernel — element-wise max(0, x)
// Positive values pass
// Negative values become zero

__global__ void reluActivation(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        data[i] = fmaxf(0.0f, data[i]);
    }
}

int main() {

    int batchSize  = 64;
    int inputSize  = 128;
    int outputSize = 64;

    printf("Batch size :   %d\n", batchSize);
    printf("Input size :   %d\n", inputSize);
    printf("Output size :  %d\n\n", outputSize);

    size_t inputBytes  = batchSize  * inputSize  * sizeof(float);
    size_t weightBytes = inputSize  * outputSize * sizeof(float);
    size_t biasBytes   = outputSize * sizeof(float);
    size_t outputBytes = batchSize  * outputSize * sizeof(float);

    float* h_input   = (float*)malloc(inputBytes);
    float* h_weights = (float*)malloc(weightBytes);
    float* h_bias    = (float*)malloc(biasBytes);
    float* h_output  = (float*)malloc(outputBytes);

    // All ones — expected output = inputSize = 128
    for (int i = 0; i < batchSize * inputSize;  i++) h_input[i]   = 1.0f;
    for (int i = 0; i < inputSize * outputSize; i++) h_weights[i] = 1.0f;
    for (int i = 0; i < outputSize;             i++) h_bias[i]    = 0.0f;

    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input,   inputBytes);
    cudaMalloc(&d_weights, weightBytes);
    cudaMalloc(&d_bias,    biasBytes);
    cudaMalloc(&d_output,  outputBytes);

    cudaMemcpy(d_input,   h_input,   inputBytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weightBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias,    h_bias,    biasBytes,   cudaMemcpyHostToDevice);

    // 2D launch — one thread per output element
    dim3 fcBlock(16, 16);
    dim3 fcGrid((outputSize + 15) / 16, (batchSize + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // FC forward pass 
    cudaEventRecord(start);
    fcForward<<<fcGrid, fcBlock>>>(d_input, d_weights, d_bias,
                                    d_output,
                                    batchSize, inputSize, outputSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float fcMs = 0;
    cudaEventElapsedTime(&fcMs, start, stop);

    cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost);

    printf("FC Forward Pass:\n");
    printf("  Time: %.3f ms\n", fcMs);
    printf("  output[0][0] = %.1f (expected %.1f)\n",
           h_output[0], (float)inputSize);
    printf("  output[0][1] = %.1f (expected %.1f)\n",
           h_output[1], (float)inputSize);

    // ---- ReLU check ----

    float h_test[] = {-5.0f, 3.0f, -1.0f, 7.0f, 0.0f, -2.0f};
    float *d_test;
    cudaMalloc(&d_test, 6 * sizeof(float));
    cudaMemcpy(d_test, h_test, 6 * sizeof(float), cudaMemcpyHostToDevice);

    reluActivation<<<1, 32>>>(d_test, 6);
    cudaDeviceSynchronize();

    float h_relu_out[6];
    cudaMemcpy(h_relu_out, d_test, 6 * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("ReLU Demonstration:\n");
    printf("  Input:  [-5, 3, -1, 7, 0, -2]\n");
    printf("  Output: [%.0f, %.0f, %.0f, %.0f, %.0f, %.0f]\n",
           h_relu_out[0], h_relu_out[1], h_relu_out[2],
           h_relu_out[3], h_relu_out[4], h_relu_out[5]);
    printf("  Expected: [0, 3, 0, 7, 0, 0]\n");

    // Apply ReLU to actual FC output
    int totalElements = batchSize * outputSize;
    reluActivation<<<(totalElements + 255) / 256, 256>>>(
        d_output, totalElements);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost);
    printf("\nReLU on FC output:\n");
    printf("  output[0][0] after ReLU = %.1f",  h_output[0]);
    printf(" (positive — unchanged)\n");

    cudaFree(d_input); cudaFree(d_weights);
    cudaFree(d_bias);  cudaFree(d_output); cudaFree(d_test);
    free(h_input); free(h_weights); free(h_bias); free(h_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
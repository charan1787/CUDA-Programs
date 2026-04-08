
// Softmax Steps :
//   1. Find max value    
//   2. Subtract max     
//   3. Exponentiate      
//   4. Sum all values   
//   5. Divide by sum     
 

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Softmax kernel for one vector of length N
// Uses shared memory reduction for max and sum

__global__ void softmaxForward(float* input, float* output, int N) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Find max value 
    sdata[tid] = (i < N) ? input[i] : -1e38f;
    __syncthreads();

    // Max reduction 
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    // Subtract max and exponentiate
    // Subtracting max prevents exp() from overflowing to infinity
    // exp(z[i] - max) is always <= exp(0) = 1.0
    float val = (i < N) ? expf(input[i] - max_val) : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    // Sum reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = sdata[0];

    // Divide by sum
    if (i < N) {
        output[i] = val / sum_val;
    }
}

int main() {
   
    // Small example from Textbook
    int N = 4;
    float h_input[]  = {2.0f, 1.0f, 0.5f, -1.0f};
    float h_output[4] = {0};

    printf("Test 1 :\n");
    printf("Input logits: [2.0, 1.0, 0.5, -1.0]\n\n");

    float *d_input, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float),
               cudaMemcpyHostToDevice);

    softmaxForward<<<1, 256>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("Softmax output (probabilities):\n");
    float sum = 0;
    for (int i = 0; i < N; i++) {
        printf("  output[%d] = %.6f\n", i, h_output[i]);
        sum += h_output[i];
    }
    printf(" \n Sum = %.6f (expected 1.0)\n\n", sum);


    // Test 2 Uniform input — all equal probabilities
    
    int N2 = 64;
    float* h_uniform = (float*)malloc(N2 * sizeof(float));
    float* h_out2    = (float*)malloc(N2 * sizeof(float));

    for (int i = 0; i < N2; i++) h_uniform[i] = 1.0f;

    float *d_uniform, *d_out2;
    cudaMalloc(&d_uniform, N2 * sizeof(float));
    cudaMalloc(&d_out2,    N2 * sizeof(float));

    cudaMemcpy(d_uniform, h_uniform, N2 * sizeof(float),
               cudaMemcpyHostToDevice);

    softmaxForward<<<1, 256>>>(d_uniform, d_out2, N2);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out2, d_out2, N2 * sizeof(float),
               cudaMemcpyDeviceToHost);

    float sum2 = 0;
    for (int i = 0; i < N2; i++) sum2 += h_out2[i];

    printf("Test 2 :\n Uniform input (64 elements all = 1.0):\n");
    printf("  Expected: each output = 1/64 = %.6f\n", 1.0f/N2);
    printf("  output[0] = %.6f\n", h_out2[0]);
    printf("  output[1] = %.6f\n", h_out2[1]);
    printf("  Sum = %.6f (expected 1.0)\n", sum2);
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_uniform); cudaFree(d_out2);
    free(h_uniform); free(h_out2);

    return 0;
}
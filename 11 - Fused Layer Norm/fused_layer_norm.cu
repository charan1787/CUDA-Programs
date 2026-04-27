#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Layer norm equation:
//   mean     = sum(x) / N
//   variance = sum((x - mean)^2) / N
//   output   = gamma * (x - mean) / sqrt(variance + epsilon) + beta

#define THREADS 256
#define EPSILON 1e-5f


__global__ void layerNormUnfused(float* input, float* output,
                                  float* gamma, float* beta,
                                  float* temp_mean, float* temp_var,
                                  int N) {
    __shared__ float s[THREADS];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Pass 1 — compute mean
    float val = (i < N) ? input[i] : 0.0f;
    s[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    // WRITE mean to global memory — simulates separate kernel boundary
    if (tid == 0) temp_mean[blockIdx.x] = s[0] / N;
    __syncthreads();

    // Pass 2 — compute variance (READ mean from global memory)
    float mean = temp_mean[blockIdx.x];    // READ from global memory
    s[tid] = (i < N) ? (val - mean) * (val - mean) : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    // WRITE variance to global memory — another separate kernel boundary
    if (tid == 0) temp_var[blockIdx.x] = s[0] / N;
    __syncthreads();

    // Pass 3 — normalise (READ variance from global memory)
    float variance = temp_var[blockIdx.x]; // READ from global memory
    if (i < N) {
        float norm = (val - mean) / sqrtf(variance + EPSILON);
        output[i]  = gamma[i] * norm + beta[i];
    }
}

__global__ void layerNormFused(float* input, float* output,
                                float* gamma, float* beta,
                                int N) {
    __shared__ float s[THREADS];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // GLOBAL READ 1: load input into register and shared memory
    float val = (i < N) ? input[i] : 0.0f;
    s[tid] = val;
    __syncthreads();

    // Step 1: Sum reduction for mean — stays in shared memory
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    // mean is in a REGISTER — zero global memory cost
    float mean = s[0] / N;
    __syncthreads();

    // Step 2: Variance reduction — val and mean both in registers
    // No global memory needed — reuse shared memory scratchpad
    s[tid] = (i < N) ? (val - mean) * (val - mean) : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    // variance is in a REGISTER — zero global memory cost
    float variance = s[0] / N;

    // Step 3: Normalise, scale, shift
    // val, mean, variance all in registers — only reads gamma and beta
    if (i < N) {
        float norm = (val - mean) / sqrtf(variance + EPSILON);
        // GLOBAL WRITE: only write at the very end
        output[i] = gamma[i] * norm + beta[i];
    }
}

int main() {
    int N = 1 << 20;   // 1 million elements
    size_t size = N * sizeof(float);

    // Host memory
    float* h_input   = (float*)malloc(size);
    float* h_gamma   = (float*)malloc(size);
    float* h_beta    = (float*)malloc(size);
    float* h_unfused = (float*)malloc(size);
    float* h_fused   = (float*)malloc(size);

    // Initialise input with varied values
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 256) - 128.0f;  // range -128 to 127
        h_gamma[i] = 1.0f;   // no scaling
        h_beta[i]  = 0.0f;   // no shift
    }

    // Device memory
    float *d_input, *d_output_unfused, *d_output_fused;
    float *d_gamma, *d_beta;
    float *d_temp_mean, *d_temp_var;

    int blocks = (N + THREADS - 1) / THREADS;

    cudaMalloc(&d_input,          size);
    cudaMalloc(&d_output_unfused, size);
    cudaMalloc(&d_output_fused,   size);
    cudaMalloc(&d_gamma,          size);
    cudaMalloc(&d_beta,           size);
    cudaMalloc(&d_temp_mean,      blocks * sizeof(float));
    cudaMalloc(&d_temp_var,       blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int runs = 100;

    // Benchmark for unfused
    float totalUnfused = 0;
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        layerNormUnfused<<<blocks, THREADS>>>(
            d_input, d_output_unfused,
            d_gamma, d_beta,
            d_temp_mean, d_temp_var, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalUnfused += ms;
    }
    printf("Unfused layer norm: %.4f ms\n", totalUnfused / runs);

    // Benchmark for fused
    float totalFused = 0;
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        layerNormFused<<<blocks, THREADS>>>(
            d_input, d_output_fused,
            d_gamma, d_beta, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalFused += ms;
    }
    printf("Fused layer norm:   %.4f ms\n", totalFused / runs);
    printf("Speedup:            %.2fx\n", totalUnfused / totalFused);

    // correctness 
    cudaMemcpy(h_unfused, d_output_unfused, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fused,   d_output_fused,   size, cudaMemcpyDeviceToHost);


    // Free memory
    cudaFree(d_input);
    cudaFree(d_output_unfused); cudaFree(d_output_fused);
    cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_temp_mean); cudaFree(d_temp_var);
    free(h_input); free(h_gamma); free(h_beta);
    free(h_unfused); free(h_fused);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

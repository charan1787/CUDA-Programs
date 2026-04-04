#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused Softmax Kernel
// Unfused softmax: 5 separate kernel launches, 10 global memory accesses/element
// Fused softmax:   1 kernel launch, ~3 global memory accesses/element
// All intermediate results (max, sum) stay in shared memory and registers.
// Never written to global memory between steps.
// This is the pattern behind every fused operation in cuDNN and FlashAttention.


#define THREADS 256


// UNFUSED softmax — separate operations for comparison
// We simulate this by writing intermediate results to global arrays
// instaed of writing 5 steps (readme check) 
__global__ void softmaxUnfused(float* input, float* output,
                                float* temp_max, float* temp_sum, int N) {
    __shared__ float s[THREADS];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // Pass 1 — find max
    s[tid] = (i < N) ? input[i] : -1e38f;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid+stride]);
        __syncthreads();
    }
    if (tid == 0) temp_max[blockIdx.x] = s[0];  // WRITE to global memory
    __syncthreads();

    // Pass 2 — subtract max and exp (reads temp_max from global memory)
    float max_val = temp_max[blockIdx.x];         // READ from global memory
    float exp_val = (i < N) ? expf(input[i] - max_val) : 0.0f;

    // Pass 3 — sum reduction
    s[tid] = (i < N) ? exp_val : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid+stride];
        __syncthreads();
    }
    if (tid == 0) temp_sum[blockIdx.x] = s[0];  // WRITE to global memory
    __syncthreads();

    // Pass 4 — divide (reads temp_sum from global memory)
    float sum_val = temp_sum[blockIdx.x];         // READ from global memory
    if (i < N) output[i] = exp_val / sum_val;
}


// FUSED softmax — everything in one kernel
// Max, subtract, exp, sum, divide — all stay in registers/shared memory
// Only TWO reads from global memory and ONE write per element

__global__ void softmaxFused(float* input, float* output, int N) {
    __shared__ float s[THREADS];
    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // GLOBAL READ 1: load input into shared memory
    float val = (i < N) ? input[i] : -1e38f;
    s[tid] = val;
    __syncthreads();

    // Step 1: Max reduction — stays in shared memory, never goes to global
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] = fmaxf(s[tid], s[tid+stride]);
        __syncthreads();
    }
    float max_val = s[0];  // In a register — zero global memory cost
    __syncthreads();

    // GLOBAL READ 2: read input again, subtract max, exponentiate
    // max_val is in a register — no global memory access needed
    float exp_val = (i < N) ? expf(input[i] - max_val) : 0.0f;
    s[tid] = exp_val;
    __syncthreads();

    // Step 2: Sum reduction — stays in shared memory, never goes to global
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid+stride];
        __syncthreads();
    }
    float sum_val = s[0];  // In a register — zero global memory cost

    // GLOBAL WRITE 1: write final result — the ONLY write in this kernel
    if (i < N) output[i] = exp_val / sum_val;
}

int main() {
    int N = 1 << 20;  // 1 million elements
    size_t size = N * sizeof(float);

    // Host data
    float* h_input    = (float*)malloc(size);
    float* h_unfused  = (float*)malloc(size);
    float* h_fused    = (float*)malloc(size);

    // Initialise with values 0 to 1 (scaled)
    for (int i = 0; i < N; i++) h_input[i] = (float)(i % 256) / 256.0f;

    // Device memory
    float *d_input, *d_unfused, *d_fused;
    float *d_temp_max, *d_temp_sum;

    int blocks = (N + THREADS - 1) / THREADS;

    cudaMalloc(&d_input,    size);
    cudaMalloc(&d_unfused,  size);
    cudaMalloc(&d_fused,    size);
    cudaMalloc(&d_temp_max, blocks * sizeof(float));
    cudaMalloc(&d_temp_sum, blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int runs = 100;

    // Benchmark : unfused 
    float totalUnfused = 0;
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        softmaxUnfused<<<blocks, THREADS>>>(
            d_input, d_unfused, d_temp_max, d_temp_sum, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalUnfused += ms;
    }
    printf("Unfused softmax: %.4f ms\n", totalUnfused / runs);

    // Benchmark : fused 
    float totalFused = 0;
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        softmaxFused<<<blocks, THREADS>>>(d_input, d_fused, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalFused += ms;
    }
    printf("Fused softmax:   %.4f ms\n", totalFused / runs);
    printf("Speedup:         %.2fx\n", totalUnfused / totalFused);

    // ---- Verify outputs match ----
    cudaMemcpy(h_unfused, d_unfused, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fused,   d_fused,   size, cudaMemcpyDeviceToHost);

    // Check first block sums to 1.0
    float unfused_sum = 0, fused_sum = 0;
    for (int i = 0; i < THREADS; i++) {
        unfused_sum += h_unfused[i];
        fused_sum   += h_fused[i];
    }
    printf("\nFirst block sum (unfused): %.6f (expected 1.0)\n", unfused_sum);
    printf("First block sum (fused):   %.6f (expected 1.0)\n", fused_sum);

    // Check values match between unfused and fused
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_unfused[i] - h_fused[i]) > 1e-5f) errors++;
    }
    // Free memory 
    cudaFree(d_input); cudaFree(d_unfused); cudaFree(d_fused);
    cudaFree(d_temp_max); cudaFree(d_temp_sum);
    free(h_input); free(h_unfused); free(h_fused);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
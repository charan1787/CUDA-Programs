

 //   Implements and benchmarks 5 versions of parallel sum reduction,
 //   each one fixing a specific performance problem from the previous.
 //   Computes sum of N float elements in parallel on the GPU.

 //   Sequential sum: O(N) steps, 1 thread
 //   Parallel tree:  O(log N) steps, N/2 threads active each step
 
 //   V1 Naive:         2.9528 ms (baseline)
 //   V2 No divergence: 1.0403 ms (2.84x speedup)
 //   V3 Better load:   0.3773 ms (7.83x speedup)
 //   V4 Min global:    0.2365 ms (12.48x speedup)
 //   V5 Coarsened:     0.2441 ms (12.10x speedup)



#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS 256   

__global__ void reduction_v1(float* in, float* out, int N) {
    __shared__ float s[THREADS];   

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    s[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();   
    for (int stride = 1; stride < blockDim.x; stride *= 2) {

        if (tid % (2 * stride) == 0) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();   
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}


__global__ void reduction_v2(float* in, float* out, int N) {
    __shared__ float s[THREADS];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    s[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // NO DIVERGENCE
        if (tid < stride) {
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}

__global__ void reduction_v3(float* in, float* out, int N) {
    __shared__ float s[THREADS];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = (i < N) ? in[i] : 0.0f;
    if (i + blockDim.x < N) val += in[i + blockDim.x];
    s[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}

__global__ void reduction_v4(float* in, float* out, int N) {
    __shared__ float s[THREADS];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 4) + threadIdx.x;

    float val = 0.0f;
    for (int k = 0; k < 4; k++) {
        int idx = i + k * blockDim.x;
        if (idx < N) val += in[idx];
    }
    s[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}

__global__ void reduction_v5(float* in, float* out, int N) {
    __shared__ float s[THREADS];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 8) + threadIdx.x;

    float val = 0.0f;
    for (int k = 0; k < 8; k++) {
        int idx = i + k * blockDim.x;
        if (idx < N) val += in[idx];
    }
    s[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = s[0];
}



float benchmark(void (*kernel)(float*, float*, int),
                float* d_in, float* d_out, int N, int blocks, int runs) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total = 0;
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        kernel<<<blocks, THREADS>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total += ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total / runs;
}

int main() {
    int N = 1 << 24;   // 16,777,216 elements — large to get stable measurements
    size_t size = N * sizeof(float);

    printf("Parallel Reduction Benchmark\n");
    printf("N = %d elements (%.1f MB)\n", N, size / 1e6);
    printf("Threads per block: %d\n", THREADS);
    printf("Runs per kernel: 100\n\n");

    float* h_in = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float* d_in;
    cudaMalloc(&d_in, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int maxBlocks = (N + THREADS - 1) / THREADS;
    float* d_out;
    cudaMalloc(&d_out, maxBlocks * sizeof(float));


    int b1 = (N + THREADS     - 1) / THREADS;        
    int b2 = b1;
    int b3 = (N + THREADS * 2 - 1) / (THREADS * 2);  
    int b4 = (N + THREADS * 4 - 1) / (THREADS * 4);  
    int b5 = (N + THREADS * 8 - 1) / (THREADS * 8);  

    printf("Block counts:\n");
    printf("  V1,V2 (1 elem/thread): %d blocks\n", b1);
    printf("  V3    (2 elem/thread): %d blocks\n", b3);
    printf("  V4    (4 elem/thread): %d blocks\n", b4);
    printf("  V5    (8 elem/thread): %d blocks\n\n", b5);

    float t1 = benchmark(reduction_v1, d_in, d_out, N, b1, 100);
    float t2 = benchmark(reduction_v2, d_in, d_out, N, b2, 100);
    float t3 = benchmark(reduction_v3, d_in, d_out, N, b3, 100);
    float t4 = benchmark(reduction_v4, d_in, d_out, N, b4, 100);
    float t5 = benchmark(reduction_v5, d_in, d_out, N, b5, 100);

    printf("Results:\n");
    printf("V1 Naive:            %.4f ms  (baseline)\n",       t1);
    printf("V2 No divergence:    %.4f ms  (%.2fx vs V1)\n",    t2, t1/t2);
    printf("V3 Better load:      %.4f ms  (%.2fx vs V1)\n",    t3, t1/t3);
    printf("V4 Min global mem:   %.4f ms  (%.2fx vs V1)\n",    t4, t1/t4);
    printf("V5 Coarsened:        %.4f ms  (%.2fx vs V1)\n",    t5, t1/t5);
    printf("\nNote: V4 > V5 on T4 — memory bandwidth saturated at V4.\n");
    printf("More coarsening adds overhead without gaining bandwidth.\n");

    float* h_out = (float*)malloc(b5 * sizeof(float));
    cudaMemcpy(h_out, d_out, b5 * sizeof(float), cudaMemcpyDeviceToHost);

    float total_sum = 0;
    for (int i = 0; i < b5; i++) total_sum += h_out[i];

    printf("\nVerification:\n");
    printf("V5 computed sum: %.0f\n", total_sum);
    printf("Expected sum:    %d\n", N);
    printf("Result: %s\n", (int)total_sum == N ? "PASSED" : "FAILED");

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);

    return 0;
}

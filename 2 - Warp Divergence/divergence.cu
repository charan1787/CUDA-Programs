

 // Compares two kernels that produce identical results but
 //   with very different memory access patterns:
 //   1. DivergentKernel  — causes warp divergence
 //   2. Non-divergentkernel — eliminates divergence mathematically
 

 // MEASURED RESULT ON T4 Colab : 
 //   Divergent kernel:     ~1.175 ms
 //   Non-divergent kernel: ~0.034 ms
 //   Speedup:              ~34x
 

#include <stdio.h>
#include <cuda_runtime.h>

// KERNEL 1 : DivergentKernel
__global__ void divergentKernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (threadIdx.x % 2 == 0) {
            data[i] = data[i] * 2.0f;  // Even threads : multiply by 2
        } else {
            data[i] = data[i] * 3.0f;  // Odd threads : multiply by 3
        }
    }
}

// KERNEL 2 : NonDivergentKernel

__global__ void nonDivergentKernel(float* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {

        float multiplier = 2.0f + (threadIdx.x % 2) * 1.0f;
        data[i] = data[i] * multiplier;
    }
}

int main() {
    int N = 1 << 20;    // 1 million elements, large enough to measure accurately
    size_t size = N * sizeof(float);

    // Initialisation
    float* h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_data[i] = (float)i;

    // CPU->GPU
    float* d_data;
    cudaMalloc(&d_data, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Measuring time for execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Divergence
    float totalDiv = 0.0f;
    for (int run = 0; run < 100; run++) {
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        divergentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalDiv += ms;
    }
    printf("Divergent average:     %.4f ms\n", totalDiv / 100);

    // No DIvergence
    float totalNonDiv = 0.0f;
    for (int run = 0; run < 100; run++) {
        cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

        cudaEventRecord(start);
        nonDivergentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalNonDiv += ms;
    }
    printf("Non-divergent average: %.4f ms\n", totalNonDiv / 100);
    printf("Speedup:               %.2fx\n", totalDiv / totalNonDiv);

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    divergentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    float* h_result = (float*)malloc(size);
    cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);

    printf("\nVerification (divergent kernel):\n");
    printf("data[0] = %.1f  (expected 0.0  — 0 * 2 = 0)\n",  h_result[0]);
    printf("data[1] = %.1f  (expected 3.0  — 1 * 3 = 3)\n",  h_result[1]);
    printf("data[2] = %.1f  (expected 4.0  — 2 * 2 = 4)\n",  h_result[2]);
    printf("data[3] = %.1f  (expected 9.0  — 3 * 3 = 9)\n",  h_result[3]);

    // free up space
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);
    free(h_result);

    return 0;
}

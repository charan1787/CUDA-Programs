


// Adds two arrays A and B element-wise on the GPU.
// C[i] = A[i] + B[i] for every i from 0 to N-1.


#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float* A, float* B, float* C, int N) {

    // Each thread needs a unique ID to know which array element it handles.
     
    // blockIdx.x  = which block this thread is in (0, 1, 2, ...)
    // blockDim.x  = how many threads per block (e.g. 256)
    // threadIdx.x = this thread's position within its block (0 to 255)
    // Formula: i = blockIdx.x * blockDim.x + threadIdx.x
    

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // BOUNDS CHECK
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000;                       
    size_t size = N * sizeof(float);     

    // Host Memory 
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Matrix Initialisation
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;  
        h_B[i] = i * 2.0f;  
    }                         

    // Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);  
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // CPU -> GPU Memory copy
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("N = %d\n", N);
    printf("Threads per block = %d\n", threadsPerBlock);
    printf("Blocks per grid   = %d\n", blocksPerGrid);
    printf("Total threads     = %d (extra %d handled by bounds check)\n",
           blocksPerGrid * threadsPerBlock,
           blocksPerGrid * threadsPerBlock - N);

    // Execution Time calculation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);                                         // Start timer
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // Launch kernel
    cudaEventRecord(stop);                                          // Stop timer
    cudaEventSynchronize(stop);                                     // Wait for GPU to finish

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel time: %.3f ms\n", milliseconds);

    // GPU -> CPU memory copy
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


   // Freeup memory - Should not forget
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

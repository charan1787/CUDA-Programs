
// Adds two M x N matrices element-wise on the GPU.
// C[row][col] = A[row][col] + B[row][col]
 

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixAdd(float* A, float* B, float* C, int M, int N) {


    // Each thread needs a unique ID to know which array element it handles.
     
    // blockIdx.x  = which block this thread is in (0, 1, 2, ...)
    // blockDim.x  = how many threads per block (e.g. 256)
    // threadIdx.x = this thread's position within its block (0 to 255)
    // Formula: i = blockIdx.x * blockDim.x + threadIdx.x
     

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 2D BOUNDS CHECK

    if (row < M && col < N) {
        // Row-major flat indexing: idx = row * N + col
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int M = 1000;   // rows
    int N = 1000;   // columns
    size_t size = M * N * sizeof(float);

    // Host Memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Matrix Initialisation
    for (int i = 0; i < M * N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Device Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);

    printf("Grid size: x=%d, y=%d\n", gridSize.x, gridSize.y);
    printf("Block size: x=%d, y=%d\n", blockSize.x, blockSize.y);
    printf("Total threads: %d x %d = %d\n",
           gridSize.x * blockSize.x,
           gridSize.y * blockSize.y,
           gridSize.x * blockSize.x * gridSize.y * blockSize.y);

    // Execution Time calculation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %.3f ms\n", ms);

    // Output copy to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("\nVerification:\n");
    printf("C[0]    = %.1f  (expected 0.0)\n",    h_C[0]);
    printf("C[1]    = %.1f  (expected 3.0)\n",    h_C[1]);
    printf("C[1000] = %.1f  (expected 3000.0)\n", h_C[1000]);


    // memory Cleaning : SHould not miss
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C);
    free(h_A); 
    free(h_B); 
    free(h_C);
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);

    return 0;
}

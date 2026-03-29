

 // Implements and benchmarks two versions of matrix multiplication:
 //   1. naiveMatMul  — reads directly from global memory every time
 //   2. tiledMatMul  — uses shared memory to cache and reuse data


 // Global memory latency — 400-800 cycles per access
 // Shared memory — on-chip, ~5 cycle latency, cooperative
 // Tiling — load once into shared memory, reuse many times
 // __syncthreads() — two critical sync points
 // Boundary checking for non-square, non-tile-multiple matrices

 
 // MEASURED RESULTS ON T4 (512x512 matrices):
 //   Naive matmul:  ~24.555 ms
 //   Tiled matmul:  ~0.743 ms
 //   Speedup:       ~47.92x
 
 // 47x SPEEDUP?
 //   1. 32x reduction in global memory traffic (32 tiles of 16 = 512)
 //   2. Additional speedup from improved memory coalescing
 //   3. Shared memory bandwidth >> global memory bandwidth
 



#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16   // Each tile is 16x16 elements
    
// Naive approach
__global__ void naiveMatMul(float* A, float* B, float* C, int N) {
   
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;


        // These are 400-800 cycle latency (Global Memory)
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// tiled matrix
__global__ void tiledMatMul(float* A, float* B, float* C, int N) {


    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;  

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {


        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < N && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 512;
    size_t size = N * N * sizeof(float);

    
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE);

    printf("Matrix size: %d x %d\n", N, N);
    printf("Tile size:   %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Grid size:   %d x %d blocks\n", gridSize.x, gridSize.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naiveMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naiveMs = 0;
    cudaEventElapsedTime(&naiveMs, start, stop);
    printf("\nNaive matmul:  %.3f ms\n", naiveMs);

    cudaEventRecord(start);
    tiledMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiledMs = 0;
    cudaEventElapsedTime(&tiledMs, start, stop);
    printf("Tiled matmul:  %.3f ms\n", tiledMs);
    printf("Speedup:       %.2fx\n", naiveMs / tiledMs);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nVerification (all elements should equal %d):\n", N);
    printf("C[0][0] = %.1f (expected %d)\n", h_C[0], N);
    printf("C[0][1] = %.1f (expected %d)\n", h_C[1], N);

    int errors = 0;
    for (int i = 0; i < N * N; i++) {
        if (h_C[i] != (float)N) errors++;
    }
    printf("Full check: %s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

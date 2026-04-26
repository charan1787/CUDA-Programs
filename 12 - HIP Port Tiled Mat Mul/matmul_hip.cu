

// HIP runtime header — replaces cuda_runtime.h
#include <hip/hip_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

// NAIVE matmul kernel — identical in CUDA and HIP

__global__ void naiveMatMul(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


// TILED matmul kernel — identical in CUDA and HIP

__global__ void tiledMatMul(float* A, float* B, float* C, int N) {
    // Shared memory — identical syntax in CUDA and HIP
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tile into shared memory — identical to CUDA
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            tileA[threadIdx.y][threadIdx.x] =
                A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < N && col < N)
            tileB[threadIdx.y][threadIdx.x] =
                B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // __syncthreads() — identical in CUDA and HIP
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

    printf("Tiled MatMul — HIP Port \n\n");
    printf("Demonstrating CUDA to HIP translation:\n");
    printf("  cudaMalloc    -> hipMalloc\n");
    printf("  cudaMemcpy    -> hipMemcpy\n");
    printf("  cudaFree      -> hipFree\n");
    printf("  cudaEvent_t   -> hipEvent_t\n");
    printf("  Kernel code   -> IDENTICAL (zero changes)\n\n");

    // Host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Device memory  hipMalloc instead of cudaMalloc
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

    // Memory copy  hipMemcpy instead of cudaMemcpy
    // hipMemcpyHostToDevice instead of cudaMemcpyHostToDevice
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE,
                  (N + TILE_SIZE - 1) / TILE_SIZE);

    // Timing hipEvent_t instead of cudaEvent_t
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Naive matmul
    // hipEventRecord takes extra stream argument (0 = default stream)
    // cudaEventRecord does NOT require this argument
    hipEventRecord(start, 0);
    naiveMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    float naiveMs = 0;
    hipEventElapsedTime(&naiveMs, start, stop);
    printf("Naive matmul:  %.3f ms\n", naiveMs);

    // Tiled matmul 
    hipEventRecord(start, 0);
    tiledMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    float tiledMs = 0;
    hipEventElapsedTime(&tiledMs, start, stop);
    printf("Tiled matmul:  %.3f ms\n", tiledMs);
    printf("Speedup:       %.2fx\n", naiveMs / tiledMs);

    // Copy result back hipMemcpyDeviceToHost
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    // Free memory hipFree instead of cudaFree
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    free(h_A); free(h_B); free(h_C);
    hipEventDestroy(start); hipEventDestroy(stop);

    return 0;
}

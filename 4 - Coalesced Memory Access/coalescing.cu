

 // Benchmarks two kernels that copy a matrix:
 //   1. rowCopy — reads rows sequentially (coalesced access)
 //   2. colCopy — reads columns (uncoalesced, strided access)
 //   Both do identical work on the same data. Only the access pattern differs.


 // MEASURED RESULT ON T4 (4096x4096 matrix):
 //   Coalesced (row copy):   ~0.5762 ms
 //   Uncoalesced (col copy): ~1.2177 ms
 //   Slowdown:               ~2.11x


#include <stdio.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void rowCopy(float* in, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {

        out[row * cols + col] = in[row * cols + col];
    }
}


__global__ void colCopy(float* in, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {

        out[row * cols + col] = in[col * cols + row];
    }
}

int main() {
    int rows = 4096;
    int cols = 4096;
    int N = rows * cols;
    size_t size = N * sizeof(float);

    printf("Matrix size: %d x %d = %d elements\n", rows, cols, N);
    printf("Data size:   %.1f MB\n", size / 1e6);

    float *h_in = (float*)malloc(size);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE, TILE);
    dim3 gridSize((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    rowCopy<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
    colCopy<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
    cudaDeviceSynchronize();

    float totalRow = 0;
    for (int r = 0; r < 100; r++) {
        cudaEventRecord(start);
        rowCopy<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalRow += ms;
    }
    printf("\nCoalesced (row) average:    %.4f ms\n", totalRow / 100);

    float totalCol = 0;
    for (int r = 0; r < 100; r++) {
        cudaEventRecord(start);
        colCopy<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalCol += ms;
    }
    printf("Uncoalesced (col) average:  %.4f ms\n", totalCol / 100);
    printf("Slowdown from uncoalescing: %.2fx\n", totalCol / totalRow);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

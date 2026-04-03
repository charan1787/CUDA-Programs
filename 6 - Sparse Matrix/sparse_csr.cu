
// Sparse matrix A (4 * 4):
// 1 0 0 2
// 0 3 0 0
// 0 0 4 0
// 0 0 0 5

#include <stdio.h>
#include <cuda_runtime.h>

// SpMV CSR kernel 
__global__ void spmvCSR(float* values, int* colIdx, int* rowPtr,
                          float* x, float* y, int numRows) {

    // Each thread handles one row
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        float sum = 0.0f;

        // rowPtr[row] tells us where this row's non-zeros start
        // rowPtr[row+1] tells us where the next row starts
        // So this row's non-zeros are at indices rowPtr[row] to rowPtr[row+1]-1
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; j++) {
            // values[j] is the non-zero element
            // colIdx[j] tells us which element of x to multiply with
            sum += values[j] * x[colIdx[j]];
        }

        y[row] = sum;
    }
}

int main() {
    printf("=== SpMV with CSR Format ===\n\n");

    // CSR representation of our sparse matrix - Texbook Example
    int numRows     = 4;
    int numNonZeros = 5;

    float h_values[] = {1, 2, 3, 4, 5};  // Non-zero values row by row
    int   h_colIdx[] = {0, 3, 1, 2, 3};  // Column of each non-zero
    int   h_rowPtr[] = {0, 2, 3, 4, 5};  // Where each row starts

    float h_x[] = {1, 1, 1, 1};  // Input vector — all ones
    float h_y[4] = {0};           // Output vector

    printf("Sparse matrix A : \n");
    printf("1 0 0 2\n");
    printf("0 3 0 0\n");
    printf("0 0 4 0\n");
    printf("0 0 0 5\n\n");

    printf("CSR representation:\n");
    printf("values = [1, 2, 3, 4, 5]\n");
    printf("colIdx = [0, 3, 1, 2, 3]\n");
    printf("rowPtr = [0, 2, 3, 4, 5]\n\n");

    printf("Input vector x = [1, 1, 1, 1]\n\n");

    // Device memory
    float *d_values, *d_x, *d_y;
    int   *d_colIdx, *d_rowPtr;

    cudaMalloc(&d_values, numNonZeros * sizeof(float));
    cudaMalloc(&d_colIdx, numNonZeros * sizeof(int));
    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_x,      numRows * sizeof(float));
    cudaMalloc(&d_y,      numRows * sizeof(float));

    cudaMemcpy(d_values, h_values, numNonZeros * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, numNonZeros * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, h_rowPtr, (numRows + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, numRows * sizeof(float),
               cudaMemcpyHostToDevice);

    // One thread per row
    spmvCSR<<<1, 32>>>(d_values, d_colIdx, d_rowPtr, d_x, d_y, numRows);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, numRows * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result y = A * x:\n");
    printf("y[0] = %.1f  (expected 3.0)\n", h_y[0]);
    printf("y[1] = %.1f  (expected 3.0)\n", h_y[1]);
    printf("y[2] = %.1f  (expected 4.0)\n", h_y[2]);
    printf("y[3] = %.1f  (expected 5.0)\n", h_y[3]);

    cudaFree(d_values); cudaFree(d_colIdx);
    cudaFree(d_rowPtr); cudaFree(d_x); cudaFree(d_y);

    return 0;
}
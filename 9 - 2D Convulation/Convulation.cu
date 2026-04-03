
// 2D Convolution with Constant Memory
 
// Applies a 3x3 filter to a 2D input image.
// Filter stored in constant memory — read only.

#include <stdio.h>
#include <cuda_runtime.h>

#define FILTER_SIZE 3
#define FILTER_RADIUS 1  // 3x3 filter extends 1 pixel in each direction

// Filter stored in constant memory
// Read-only, perfect for convolution filters
// Every thread reads the same filter values
__constant__ float d_filter[FILTER_SIZE][FILTER_SIZE];


 // Zero padding — out of bounds input treated as 0
 
__global__ void conv2D(float* input, float* output,
                        int height, int width) {

    // thread's output position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        float sum = 0.0f;

        // Apply 3x3 filter 
        for (int fr = 0; fr < FILTER_SIZE; fr++) {
            for (int fc = 0; fc < FILTER_SIZE; fc++) {

                // Input position that this filter element covers
                int inputRow = row + fr - FILTER_RADIUS;
                int inputCol = col + fc - FILTER_RADIUS;

                // Zero padding so skipping out of bounds
                if (inputRow >= 0 && inputRow < height &&
                    inputCol >= 0 && inputCol < width) {

                    // d_filter reads from constant memory
                    sum += input[inputRow * width + inputCol]
                         * d_filter[fr][fc];
                }
            }
        }

        output[row * width + col] = sum;
    }
}

int main() {
    printf("2D Convolution with Constant Memory \n\n");

    int height = 8;
    int width  = 8;
    int N      = height * width;

    // Test 1 : Edge detection filter on uniform input 
    float h_filter[FILTER_SIZE][FILTER_SIZE] = {
        { 1,  0, -1},
        { 2,  0, -2},
        { 1,  0, -1}
    };

    printf("Filter (horizontal edge detection):\n");
    printf("  1  0 -1\n");
    printf("  2  0 -2\n");
    printf("  1  0 -1\n\n");

    // Copy filter to constant memory
    cudaMemcpyToSymbol(d_filter, h_filter,
                        FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // Input image — all ones (uniform, no edges)
    float* h_input  = (float*)malloc(N * sizeof(float));
    float* h_output = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input,  N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float),
               cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width  + 15) / 16,
                  (height + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv2D<<<gridSize, blockSize>>>(d_input, d_output, height, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_output, d_output, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("Input image (8x8, all ones):\n");
    for (int r = 0; r < height; r++) {
        printf("  ");
        for (int c = 0; c < width; c++) {
            printf("%.0f ", h_input[r * width + c]);
        }
        printf("\n");
    }

    printf("\nConvolution output:\n");
    for (int r = 0; r < height; r++) {
        printf("  ");
        for (int c = 0; c < width; c++) {
            printf("%5.1f ", h_output[r * width + c]);
        }
        printf("\n");
    }
    printf("\nConvolution time: %.3f ms\n", ms);
    printf("Interior values = 0.0 (no edges in uniform image)\n\n");

    // Test 2 : Smoothing filter on non-uniform input
    float h_filter2[FILTER_SIZE][FILTER_SIZE] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    printf("Filter 2 (Gaussian smoothing, sum=16):\n");
    printf("  1 2 1\n");
    printf("  2 4 2\n");
    printf("  1 2 1\n\n");

    cudaMemcpyToSymbol(d_filter, h_filter2,
                        FILTER_SIZE * FILTER_SIZE * sizeof(float));

    // Non-uniform input, values equal their index
    for (int i = 0; i < N; i++) h_input[i] = (float)(i % 8 + 1);

    cudaMemcpy(d_input, h_input, N * sizeof(float),
               cudaMemcpyHostToDevice);

    conv2D<<<gridSize, blockSize>>>(d_input, d_output, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("Input image (values 1-8 repeating):\n");
    for (int r = 0; r < height; r++) {
        printf("  ");
        for (int c = 0; c < width; c++) {
            printf("%5.0f ", h_input[r * width + c]);
        }
        printf("\n");
    }

    printf("\nSmoothed output:\n");
    for (int r = 0; r < height; r++) {
        printf("  ");
        for (int c = 0; c < width; c++) {
            printf("%6.1f ", h_output[r * width + c]);
        }
        printf("\n");
    }

    printf("\nManual verification output[1][1]:\n");
    printf("  = 1*1+2*2+1*3 + 2*1+4*2+2*3 + 1*1+2*2+1*3\n");
    printf("  = 1+4+3 + 2+8+6 + 1+4+3 = 32\n");
    printf("  Computed: %.1f %s\n",
           h_output[1 * width + 1],
           h_output[1 * width + 1] == 32.0f ? "✓" : "✗");

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
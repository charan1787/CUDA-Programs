#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// flash_attention.cu
// Simplified FlashAttention

#define MAX_SEQ_LEN 64   // max sequence length for register arrays
#define BLOCK_SIZE  16   // tile size — keys processed per iteration


// NAIVE attention
// Computes full [N x N] score matrix in registers

__global__ void naiveAttention(float* Q, float* K, float* V,
                                float* output, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // Step 1 : compute all attention scores for query 'row'
    // scores[j] = dot(Q[row], K[j]) / sqrt(d)
    float scores[MAX_SEQ_LEN];
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++)
            dot += Q[row * d + k] * K[j * d + k];
        scores[j] = dot / sqrtf((float)d);
    }

    // Step 2 : softmax over all scores — needs full row visible
    float max_val = scores[0];
    for (int j = 1; j < N; j++)
        max_val = fmaxf(max_val, scores[j]);

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        scores[j] = expf(scores[j] - max_val);
        sum += scores[j];
    }
    for (int j = 0; j < N; j++)
        scores[j] /= sum;

    // Step 3 : weighted sum of V
    for (int k = 0; k < d; k++) {
        float out = 0.0f;
        for (int j = 0; j < N; j++)
            out += scores[j] * V[j * d + k];
        output[row * d + k] = out;
    }
}

// FLASH ATTENTION — tiled with online softmax

__global__ void flashAttention(float* Q, float* K, float* V,
                                float* output, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // online softmax
    float running_max = -1e38f;  // tracks maximum score so far
    float running_sum = 0.0f;    // tracks normalisation denominator

    // Output accumulator for weighted sum of V vectors
    // Starts at zero, built up tile by tile
    float acc[8] = {0.0f};  // d=8 for our example

    // Process K and V in tiles of BLOCK_SIZE
    // This is the key loop — never process all N at once
    for (int tile = 0; tile < N; tile += BLOCK_SIZE) {

        // Step 1 : compute scores for this tile only
        // tile_scores lives in registers, never in global memory
        float tile_scores[BLOCK_SIZE];
        for (int j = 0; j < BLOCK_SIZE && (tile + j) < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[row * d + k] * K[(tile + j) * d + k];
            tile_scores[j] = dot / sqrtf((float)d);
        }

        // Step 2 : find max in this tile
        float tile_max = tile_scores[0];
        for (int j = 1; j < BLOCK_SIZE && (tile + j) < N; j++)
            tile_max = fmaxf(tile_max, tile_scores[j]);

        // Step 3 : update running max
        float new_max = fmaxf(running_max, tile_max);

        // Step 4 : ONLINE SOFTMAX CORRECTION (*)
        // Heart of FlashAttention
        // exp(old_max - new_max) < 1 always — rescales values down
        float correction = expf(running_max - new_max);
        running_sum *= correction;          // rescale denominator
        for (int k = 0; k < d; k++)
            acc[k] *= correction;           // rescale accumulated output

        // Step 5 : add this tile's contribution
        for (int j = 0; j < BLOCK_SIZE && (tile + j) < N; j++) {
            float exp_score = expf(tile_scores[j] - new_max);
            running_sum += exp_score;
            for (int k = 0; k < d; k++)
                acc[k] += exp_score * V[(tile + j) * d + k];
        }

        // Update running max for next tile
        running_max = new_max;
    }

    // Step 6 : normalise and write output
    // This is the ONLY write to global memory per output element
    for (int k = 0; k < d; k++)
        output[row * d + k] = acc[k] / running_sum;
}

int main() {
    // Small example — N=64 tokens, d=8 head dimension
    // Small enough to verify correctness manually
    int N = 64;
    int d = 8;
    size_t mat_size = N * d * sizeof(float);

    printf("Simplified FlashAttention\n\n");
    printf("Sequence length N = %d\n", N);
    printf("Head dimension  d = %d\n\n", d);

    // Host memory creation
    float* h_Q      = (float*)malloc(mat_size);
    float* h_K      = (float*)malloc(mat_size);
    float* h_V      = (float*)malloc(mat_size);
    float* h_naive  = (float*)malloc(mat_size);
    float* h_flash  = (float*)malloc(mat_size);

    // Initialise with small random-like values
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)(i % 7) - 3.0f) / 10.0f;
        h_K[i] = ((float)(i % 5) - 2.0f) / 10.0f;
        h_V[i] = ((float)(i % 11) - 5.0f) / 10.0f;
    }

    // Device memory creation
    float *d_Q, *d_K, *d_V, *d_naive, *d_flash;
    cudaMalloc(&d_Q,     mat_size);
    cudaMalloc(&d_K,     mat_size);
    cudaMalloc(&d_V,     mat_size);
    cudaMalloc(&d_naive, mat_size);
    cudaMalloc(&d_flash, mat_size);

    cudaMemcpy(d_Q, h_Q, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, mat_size, cudaMemcpyHostToDevice);

    // One thread per query token
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Naive attention 
    float totalNaive = 0;
    for (int r = 0; r < 100; r++) {
        cudaEventRecord(start);
        naiveAttention<<<blocks, threads>>>(d_Q, d_K, d_V, d_naive, N, d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalNaive += ms;
    }
    printf("Naive attention:  %.4f ms\n", totalNaive / 100);

    // Flash attention
    float totalFlash = 0;
    for (int r = 0; r < 100; r++) {
        cudaEventRecord(start);
        flashAttention<<<blocks, threads>>>(d_Q, d_K, d_V, d_flash, N, d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalFlash += ms;
    }
    printf("Flash attention:  %.4f ms\n", totalFlash / 100);
    printf("Speedup:          %.2fx\n\n", totalNaive / totalFlash);

    // Verify outputs match
    cudaMemcpy(h_naive, d_naive, mat_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flash, d_flash, mat_size, cudaMemcpyDeviceToHost);


    // Show first few output values
    printf("First output token (naive):  ");
    for (int k = 0; k < d; k++) printf("%.4f ", h_naive[k]);
    printf("\nFirst output token (flash):  ");
    for (int k = 0; k < d; k++) printf("%.4f ", h_flash[k]);
    printf("\n\n");

    // Memory comparison
    printf("Memory usage comparison:\n");
    printf("  Naive :  O(N^2) = %d floats = %.2f KB\n",
            N*N, N*N*4.0f/1024);
    printf("  Flash:  O(N)  = tile registers only\n");
    printf("\nFor N=4096 (for LLM's):\n");
    printf("  Naive :  4096 x 4096 x 4 = %.1f MB per attention head\n",
           4096.0f * 4096.0f * 4.0f / (1024*1024));
    printf("  Flash :  tile size only, it scales to any sequence length\n");

    // Free memory
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_naive); cudaFree(d_flash);
    free(h_Q); free(h_K); free(h_V);
    free(h_naive); free(h_flash);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
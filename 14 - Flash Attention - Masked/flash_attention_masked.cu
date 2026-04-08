#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>


// Fused Attention with Causal Masking


#define BLOCK_SIZE 16

// FULL attention — no causal mask - only for comparision of output - same kernel as 13 flash attention

__global__ void flashAttentionFull(float* Q, float* K, float* V,
                                    float* output, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float running_max = -1e38f;
    float running_sum = 0.0f;
    float acc[8] = {0.0f};

    for (int tile = 0; tile < N; tile += BLOCK_SIZE) {
        float tile_scores[BLOCK_SIZE];

        // Computing scores for this tile
        for (int j = 0; j < BLOCK_SIZE && (tile + j) < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++)
                dot += Q[row * d + k] * K[(tile + j) * d + k];
            tile_scores[j] = dot / sqrtf((float)d);
        }

        // Finding tile max
        float tile_max = tile_scores[0];
        for (int j = 1; j < BLOCK_SIZE && (tile + j) < N; j++)
            tile_max = fmaxf(tile_max, tile_scores[j]);

        // Update running max and correct accumulated values
        float new_max = fmaxf(running_max, tile_max);
        float correction = expf(running_max - new_max);
        running_sum *= correction;
        for (int k = 0; k < d; k++)
            acc[k] *= correction;

        // Accumulating this tile
        for (int j = 0; j < BLOCK_SIZE && (tile + j) < N; j++) {
            float exp_score = expf(tile_scores[j] - new_max);
            running_sum += exp_score;
            for (int k = 0; k < d; k++)
                acc[k] += exp_score * V[(tile + j) * d + k];
        }

        running_max = new_max;
    }

    for (int k = 0; k < d; k++)
        output[row * d + k] = acc[k] / running_sum;
}

// CAUSAL attention — with lower triangular mask


__global__ void flashAttentionCausal(float* Q, float* K, float* V,
                                      float* output, int N, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float running_max = -1e38f;
    float running_sum = 0.0f;
    float acc[8] = {0.0f};

    // Only process tiles up to and including current query position
    for (int tile = 0; tile <= row; tile += BLOCK_SIZE) {

        float tile_scores[BLOCK_SIZE];

        // Compute scores : apply causal mask for future positions
        for (int j = 0; j < BLOCK_SIZE && (tile + j) < N; j++) {
            if ((tile + j) > row) {
                // Future token : mask with very negative value
                // exp of it is 0 after softmax
                tile_scores[j] = -1e38f;
            } else {
                // Past or present token : compute real score
                float dot = 0.0f;
                for (int k = 0; k < d; k++)
                    dot += Q[row * d + k] * K[(tile + j) * d + k];
                tile_scores[j] = dot / sqrtf((float)d);
            }
        }

        // Find tile max — ignores masked positions
        float tile_max = -1e38f;
        for (int j = 0; j < BLOCK_SIZE && (tile + j) < N; j++)
            if ((tile + j) <= row)
                tile_max = fmaxf(tile_max, tile_scores[j]);

        // Update running max and correct accumulated values
        float new_max = fmaxf(running_max, tile_max);
        float correction = expf(running_max - new_max);
        running_sum *= correction;
        for (int k = 0; k < d; k++)
            acc[k] *= correction;

        // Accumulate this tile
        for (int j = 0; j < BLOCK_SIZE && (tile + j) < N; j++) {
            float exp_score = expf(tile_scores[j] - new_max);
            running_sum += exp_score;
            for (int k = 0; k < d; k++)
                acc[k] += exp_score * V[(tile + j) * d + k];
        }

        running_max = new_max;
    }

    for (int k = 0; k < d; k++)
        output[row * d + k] = acc[k] / running_sum;
}

// Reference implementation of causal attention on CPU Used to verify GPU output is correct

int main() {
    int N = 64;   // sequence length
    int d = 8;    // head dimension
    size_t size = N * d * sizeof(float);

    printf("Fused Attention with Causal Masking\n\n");
    printf("Sequence length N = %d\n", N);
    printf("Head dimension  d = %d\n\n", d);

    // Host memory creation
    float* h_Q      = (float*)malloc(size);
    float* h_K      = (float*)malloc(size);
    float* h_V      = (float*)malloc(size);
    float* h_full   = (float*)malloc(size);
    float* h_causal = (float*)malloc(size);
    float* h_cpu    = (float*)malloc(size);

    // Initialise values
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)(i % 7) - 3.0f) / 10.0f;
        h_K[i] = ((float)(i % 5) - 2.0f) / 10.0f;
        h_V[i] = ((float)(i % 11) - 5.0f) / 10.0f;
    }

    // Device memory creation
    float *d_Q, *d_K, *d_V, *d_full, *d_causal;
    cudaMalloc(&d_Q,      size);
    cudaMalloc(&d_K,      size);
    cudaMalloc(&d_V,      size);
    cudaMalloc(&d_full,   size);
    cudaMalloc(&d_causal, size);

    cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int runs = 100;

    // Full attention benchmark 

    float totalFull = 0;
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        flashAttentionFull<<<blocks, threads>>>(
            d_Q, d_K, d_V, d_full, N, d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalFull += ms;
    }
    printf("Full attention (with no mask) :   %.4f ms\n", totalFull / runs);

    //Causal attention benchmark
    float totalCausal = 0;
    for (int r = 0; r < runs; r++) {
        cudaEventRecord(start);
        flashAttentionCausal<<<blocks, threads>>>(
            d_Q, d_K, d_V, d_causal, N, d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalCausal += ms;
    }
    printf("Causal attention (with mask) :  %.4f ms\n", totalCausal / runs);
    printf("Causal speedup :   %.2fx\n\n",
           totalFull / totalCausal);

    printf("Memory usage :\n");
    printf("  Full attention :   O(N^2) without FlashAttention\n");
    printf("  Causal attention : O(N^2/2) on average (lower triangle)\n");
    printf("  Both implementations : O(N) with tiling\n");

    // Free up mem
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_full); cudaFree(d_causal); 
    free(h_Q); free(h_K); free(h_V);
    free(h_full); free(h_causal); free(h_cpu);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
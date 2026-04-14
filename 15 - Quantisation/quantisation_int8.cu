#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Per-Channel INT8 Quantisation
// Quantisation converts FP32 weights to INT8 to reduce memory:
//   FP32: 4 bytes per weight
//   INT8: 1 byte per weight — 4x compression
 
// Per-channel: each output channel gets its own scale factor
// based on its own max absolute value. Better accuracy than
// per-tensor which uses one scale for the entire matrix.
// Formula :
//   scale     = max(|weights|) / 127
//   int8_val  = round(fp32_val / scale)
//   fp32_approx = int8_val * scale

// Error = fp32_val - fp32_approx  (quantisation error)

#define THREADS 256

// Luacnh one block per output channel
// Each block:
// 1. Finds max absolute value (reduction)
// 2. Computes scale = max_abs / 127
// 3. Quantises all weights in channel

__global__ void quantiseINT8(float* input, int8_t* output,
                               float* scales,
                               int out_channels, int in_features) {
    __shared__ float sdata[THREADS];
    int channel = blockIdx.x;
    int tid     = threadIdx.x;

    if (channel >= out_channels) return;
    // Step 1: find max absolute value in this channel
    // Each thread handles multiple elements — stride pattern
    float local_max = 0.0f;
    for (int i = tid; i < in_features; i += blockDim.x) {
        local_max = fmaxf(local_max,
                          fabsf(input[channel * in_features + i]));
    } // for loop helps in less threads, more input features case.
    sdata[tid] = local_max;
    __syncthreads();

    // Max reduction — same pattern as 5 but fmaxf not addition
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    float max_abs = sdata[0];

    // Step 2: compute scale factor
    float scale = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
    if (tid == 0) scales[channel] = scale;
    __syncthreads();

    // Step 3: quantise all weights in this channel
    for (int i = tid; i < in_features; i += blockDim.x) {
        float val     = input[channel * in_features + i];
        float scaled  = val / scale;

        // Round to nearest integer
        int32_t rounded = __float2int_rn(scaled);

        // Clamp to INT8 range [-128, 127]
        rounded = max(-128, min(127, rounded));

        output[channel * in_features + i] = (int8_t)rounded;
    }
}

// DEQUANTISE kernel — INT8 back to FP32
// One block per output channel
// Simply multiply each INT8 value by the channel's scale factor

__global__ void dequantiseINT8(int8_t* input, float* output,
                                 float* scales,
                                 int out_channels, int in_features) {
    int channel = blockIdx.x;
    int tid     = threadIdx.x;

    if (channel >= out_channels) return;

    // Load this channel's scale factor
    float scale = scales[channel];

    // Dequantise: multiply INT8 by scale to get FP32 approximation
    for (int i = tid; i < in_features; i += blockDim.x) {
        output[channel * in_features + i] =
            (float)input[channel * in_features + i] * scale;
    }
}

int main() {
    // Weight matrix dimensions
    int out_channels = 64;   // number of output neurons
    int in_features  = 256;  // number of input features
    int N = out_channels * in_features;

    printf("INT8 Per-Channel Quantisation \n\n");
    printf("Weight matrix: %d channels x %d features\n",
           out_channels, in_features);
    printf("FP32 size: %.2f KB\n", N * 4.0f / 1024);
    printf("INT8 size: %.2f KB (4x compression)\n\n", N * 1.0f / 1024);

    // Host memory
    float*   h_input  = (float*)malloc(N * sizeof(float)); 
    int8_t*  h_int8   = (int8_t*)malloc(N * sizeof(int8_t));
    float*   h_dequant= (float*)malloc(N * sizeof(float));
    float*   h_scales = (float*)malloc(out_channels * sizeof(float));

    // Initialise with realistic weight values
    // Weights typically follow a normal-like distribution
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    // Device memory
    float*  d_input;
    int8_t* d_int8;
    float*  d_dequant;
    float*  d_scales;

    cudaMalloc(&d_input,   N * sizeof(float));
    cudaMalloc(&d_int8,    N * sizeof(int8_t));
    cudaMalloc(&d_dequant, N * sizeof(float));
    cudaMalloc(&d_scales,  out_channels * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaEvent_t start, stop; // just like loading a library
    cudaEventCreate(&start); // variable creation
    cudaEventCreate(&stop);

    // Quantise phase
    // One block per channel, THREADS threads per block
    cudaEventRecord(start); // variable using
    
    quantiseINT8<<<out_channels, THREADS>>>(
        d_input, d_int8, d_scales, out_channels, in_features);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float quantMs = 0;
    cudaEventElapsedTime(&quantMs, start, stop);
    printf("Quantisation time:   %.4f ms\n", quantMs);
// --------------------------------------------------------------
    // Dequantise 
    cudaEventRecord(start);

    dequantiseINT8<<<out_channels, THREADS>>>(
        d_int8, d_dequant, d_scales, out_channels, in_features);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float dequantMs = 0;
    cudaEventElapsedTime(&dequantMs, start, stop);
    printf("Dequantisation time: %.4f ms\n\n", dequantMs);

    // Copy results back
    cudaMemcpy(h_int8,   d_int8,   N * sizeof(int8_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dequant,d_dequant,N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scales, d_scales, out_channels * sizeof(float),
               cudaMemcpyDeviceToHost);

    //Accuracy analysis
    float max_error  = 0.0f;
    float mean_error = 0.0f;
    float max_val    = 0.0f;

    for (int i = 0; i < N; i++) {
        float error = fabsf(h_input[i] - h_dequant[i]);
        max_error   = fmaxf(max_error, error);
        mean_error += error;
        max_val     = fmaxf(max_val, fabsf(h_input[i]));
    }
    mean_error /= N;

    printf("Accuracy Analysis:\n");
    printf("  Max absolute error:  %.6f\n", max_error);
    printf("  Mean absolute error: %.6f\n", mean_error);
    printf("  Max weight value:    %.6f\n", max_val);
    printf("  Max error / max val: %.4f%%\n\n",
           max_error / max_val * 100);

    // first channel details
    printf("Channel 0 details :\n");
    printf("  Scale factor: %.6f\n", h_scales[0]);
    printf("  First 8 weights :\n");
    printf("  FP32 :    ");
    for (int i = 0; i < 8; i++) printf("%7.4f ", h_input[i]);
    printf("\n  INT8 :    ");
    for (int i = 0; i < 8; i++) printf("%7d ", (int)h_int8[i]);
    printf("\n  Dequant : ");
    for (int i = 0; i < 8; i++) printf("%7.4f ", h_dequant[i]);
    printf("\n  Error :   ");
    for (int i = 0; i < 8; i++)
        printf("%7.4f ", fabsf(h_input[i] - h_dequant[i]));
    printf("\n\n");

    // Memory savings
    printf("Memory Savings :\n");
    printf("  FP32 weights : %d bytes\n", N * 4);
    printf("  INT8 weights : %d bytes\n", N * 1);
    printf("  Scale factors : %d bytes\n", out_channels * 4);
    printf("  Total INT8 :   %d bytes\n",
           N * 1 + out_channels * 4);
    printf("  Compression :  %.2fx\n\n",
           (float)(N * 4) / (N * 1 + out_channels * 4));

    // Bandwidth benchmark
    // Simulate loading weights during inference
    int runs = 1000;
    float totalFP32 = 0, totalINT8 = 0;

    // FP32 weight loading simulation
    float* d_fp32_out;
    cudaMalloc(&d_fp32_out, N * sizeof(float));

    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) {
        cudaMemcpy(d_fp32_out, d_input, N * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalFP32, start, stop);

    // INT8 weight loading simulation
    int8_t* d_int8_out;
    cudaMalloc(&d_int8_out, N * sizeof(int8_t));

    cudaEventRecord(start);
    for (int r = 0; r < runs; r++) {
        cudaMemcpy(d_int8_out, d_int8, N * sizeof(int8_t),
                   cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&totalINT8, start, stop);

    printf("Weight Loading Benchmark (%d runs):\n", runs);
    printf("  FP32 loading: %.4f ms average\n", totalFP32 / runs);
    printf("  INT8 loading: %.4f ms average\n", totalINT8 / runs);
    printf("  Speedup:      %.2fx\n", totalFP32 / totalINT8);
    printf("  (INT8 moves 4x less data — directly faster)\n");
    // as the matrix is 64 * 256, it will use just registers and cache, 
    // so speed change is not much compared to DRAM and cache movement

    // delete memory space
    cudaFree(d_input); cudaFree(d_int8);
    cudaFree(d_dequant); cudaFree(d_scales);
    cudaFree(d_fp32_out); cudaFree(d_int8_out);
    free(h_input); free(h_int8);
    free(h_dequant); free(h_scales);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}

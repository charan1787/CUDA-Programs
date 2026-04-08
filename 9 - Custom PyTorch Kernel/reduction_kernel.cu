#include <torch/extension.h>
// This line imports PyTorch's C++ library. It gives us access to :
// torch::Tensor — the C++ version of a PyTorch tensor
// TORCH_CHECK — a function to validate inputs
// Everything needed to connect CUDA code to PyTorch

#define THREADS 256

// V4 reduction kernel from Day 5
// Each thread loads 4 elements, sums them,
// then shared memory tree reduction gives block sum
__global__ void reductionV4(float* input, float* output, int N) {
    __shared__ float s[THREADS];

    int tid = threadIdx.x;
    int i   = blockIdx.x * (blockDim.x * 4) + threadIdx.x;

    // Each thread loads 4 elements and adds them
    float val = 0.0f;
    for (int k = 0; k < 4; k++) {
        if (i + k * blockDim.x < N)
            val += input[i + k * blockDim.x];
    }
    s[tid] = val;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) output[blockIdx.x] = s[0];
}

// This function is called from Python
// Takes a PyTorch tensor, returns the sum as a tensor
torch::Tensor reduction_forward(torch::Tensor input) {
    int N      = input.numel(); // number of elements
    int blocks = (N + THREADS * 4 - 1) / (THREADS * 4);

    // Output array — one partial sum per block
    auto output = torch::zeros({blocks}, input.options());

    reductionV4<<<blocks, THREADS>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );

    // input.data_ptr<float>() — this is the key line. A PyTorch tensor is a high-level Python object. 
    // The GPU kernel only understands raw pointers. 
    // data_ptr<float>() extracts the raw float pointer to the tensor's underlying GPU memory. 
    // This is the bridge between the PyTorch tensor world and the raw CUDA pointer world.

    // Sum the partial results and return
    return output.sum();
}
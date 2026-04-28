import torch
import time
from torch.utils.cpp_extension import load_inline

print("Custom GELU Fusion Kernel \n")
print(f"GPU : {torch.cuda.get_device_name(0)}\n")

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// UNFUSED GELU — simulates PyTorch's two-kernel approach
// Kernel 1: compute x^3 — write to global memory
// Kernel 2: compute tanh — read from global memory
// We simulate this by doing unnecessary global memory writes

__global__ void geluUnfused(
    float* input,
    float* intermediate,  // simulates global memory write between steps
    int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = input[i];

    // Step 1 : compute x^3 and WRITE to global memory
    float x3 = x * x * x;
    intermediate[i] = x3; 
}

__global__ void geluUnfusedPart2(
    float* input,
    float* intermediate,  // READ from global memory
    float* output,
    int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x  = input[i];
    float x3 = intermediate[i];  

    float inner  = 0.7978845608f * (x + 0.044715f * x3);
    float result = 0.5f * x * (1.0f + tanhf(inner));
    output[i]    = result;
}

// FUSED GELU — everything in one kernel
// x^3 computed in register — never writte to global memory
// One read, one write — 2x less memory traffic

__global__ void geluFused(float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x      = input[i];              // ONE global read
    float x3     = x * x * x;            // in register
    float inner  = 0.7978845608f * (x + 0.044715f * x3);  // in register
    float result = 0.5f * x * (1.0f + tanhf(inner));      // in register
    output[i]    = result;                // ONE global write
}


// Wrapper for unfused version — calls two kernels back to back
// wrapper means it calls this kernel instead of default kernel when we use torch::Tensor
// later we write a using case of this torch

torch::Tensor gelu_unfused(torch::Tensor input) {
    auto intermediate = torch::empty_like(input);
    auto output       = torch::empty_like(input);
    int N       = input.numel();
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    geluUnfused<<<blocks, threads>>>(
        input.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        N
    );
    geluUnfusedPart2<<<blocks, threads>>>(
        input.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}

// Wrapper for fused version — one kernel

torch::Tensor gelu_fused(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N       = input.numel();
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    geluFused<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    return output;
}
"""
# In C++ we cannot call a function unless the compiler knows it exists first. 
# This cpp source acts as a declaration that tells the PyTorch compiler
cpp_source = """
torch::Tensor gelu_unfused(torch::Tensor input);
torch::Tensor gelu_fused(torch::Tensor input);
"""

print("Compiling the cuda kernels")
module = load_inline(
    name="gelu_fusion_v3",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["gelu_unfused", "gelu_fused"],
    extra_cuda_cflags=["-O2"],
    verbose=False
)
print("Compiled\n")

runs = 500

sizes = [
    (8,   128,  768,  "Small"),
    (8,   128,  3072, "GPT-2 FFN"),
    (16,  512,  3072, "Medium"),
    (32,  512,  3072, "Large"),
    (32,  1024, 3072, "XL"),
    (64,  1024, 3072, "XXL"),
    (128, 1024, 3072, "XXXL"),
]

# T4 L2 cache is approximately 4MB per SM
# So I assume tensors over ~32MB will likely miss cache

for batch, seq, d_ff, label in sizes:
    x = torch.randn(batch, seq, d_ff, device='cuda')
    N = x.numel()
    mb = N * 4 / 1024**2 # twice division for MB

    # Warm up
    for _ in range(10):
        _ = module.gelu_unfused(x)
        _ = module.gelu_fused(x)
    torch.cuda.synchronize()

    # Benchmark for unfused
    times_u = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = module.gelu_unfused(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_u.append((t1-t0)*1000)
    t_unfused = sum(times_u)/len(times_u)

    # Benchmark for fused
    times_f = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = module.gelu_fused(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_f.append((t1-t0)*1000)
    t_fused = sum(times_f)/len(times_f)

    speedup   = t_unfused / t_fused
    in_cache  = "Yes" if mb < 50 else "No -> DRAM"

    print(f"{label:<12} {N:<14,} {mb:<8} "
          f"{t_unfused:<14} {t_fused:<12} "
          f"{speedup:<8}x {in_cache}")

print("\n Key Insight")
print("Small tensors : fit in L2 cache -> no speedup from fusion")
print("Large tensors : hit DRAM -> fusion saves 2x memory traffic")
print("This is the roofline model in practice — same as Day 5 reduction")

# ROCm/HIP Port of Tiled Matrix Multiplication

Ported the tiled matrix multiplication kernel from CUDA to HIP
AMD's GPU programming model. Demonstrated that GPU kernel code
is 100% portable between NVIDIA and AMD hardware.

## What is HIP?
HIP (Heterogeneous Interface for Portability) is AMD's GPU
programming model — the equivalent of CUDA for AMD GPUs.
Part of the ROCm (Radeon Open Compute) platform.

AMD designed HIP to be deliberately similar to CUDA so that:
- CUDA code ports to HIP with minimal changes
- Same HIP source compiles for NVIDIA (CUDA backend) and AMD (ROCm)
- Engineers maintain one codebase for both hardware platforms

## What changes — CUDA to HIP

| CUDA | HIP |
|------|-----|
| cuda_runtime.h | hip/hip_runtime.h |
| cudaMalloc | hipMalloc |
| cudaMemcpy | hipMemcpy |
| cudaFree | hipFree |
| cudaEvent_t | hipEvent_t |
| cudaEventRecord(e) | hipEventRecord(e, 0) |
| cudaMemcpyHostToDevice | hipMemcpyHostToDevice |

## What stays IDENTICAL is the kernel code

This is exactly how HIP achieves portability
on NVIDIA. The hip* calls transparently become cuda* calls,
on AMD hardware and they use ROCm natively.

## Output : 

Naive matmul:  24.555 ms<br/>
Tiled matmul:  0.743 ms<br/>
Speedup:       47.92x<br/>

--------------
Date : 05/04/2026 <br/>
Email : charanshankar629@gmail.com


# CUDA GPU Programs

GPU AI Engineer portfolio built through a systematic curriculum
covering core CUDA programming, GPU architecture, and performance optimisation.
All benchmarks measured on NVIDIA T4 GPU (Google Colab).

---


### Thread Hierarchy
```
Kernel: vectorAdd (N=1000)

Kernel: matrixAdd (1000x1000)

```

### Warp Divergence
```
Divergent kernel:     1.1752 ms
Non-divergent kernel: 0.0338 ms
Speedup:              34x
```

### Tiled Matrix Multiplication
```
Naive matmul:  24.555 ms  (268M global memory reads)
Tiled matmul:   0.743 ms  (32 global reads per thread via shared memory)
Speedup:       47.92x
```

### Memory Coalescing
```
Matrix: 4096 x 4096

Coalesced (row copy):   0.5762 ms
Uncoalesced (col copy): 1.2177 ms
Slowdown:               2.11x
```

### Parallel Reduction
```
GPU:  NVIDIA T4
N:    16,777,216 elements

V1 Naive:            2.9528 ms  (1.00x)  — baseline, divergent
V2 No divergence:    1.0403 ms  (2.84x)  — fixed branch pattern
V3 Better load:      0.3773 ms  (7.83x)  — 2 elements per thread
V4 Min global mem:   0.2365 ms  (12.48x) — 4 elements per thread
V5 Coarsened:        0.2441 ms  (12.10x) — 8 elements (bandwidth saturated)
```

----

### GPU Architecture
- Streaming Multiprocessors (SMs), warps (32 threads), SIMT execution
- Warp divergence — causes and elimination strategies
- Occupancy — calculating from registers, shared memory, thread count
- Latency hiding — why GPUs need many warps

### Memory Hierarchy
- Registers (~1 cycle), shared memory (~5 cycles), global memory (400-800 cycles)
- Memory coalescing — consecutive threads accessing consecutive addresses
- Cache lines — 128 bytes fetched per DRAM transaction
- Tiling — cooperative loading into shared memory for data reuse
- Useful Resource - https://www.youtube.com/watch?v=Q3GgbfGTnVc

### Optimisation Framework for all kernals
```
1. Maximise occupancy
2. Coalesce global memory accesses
3. Minimise global memory traffic
4. Minimise control divergence
5. Apply thread coarsening
6. Know your bottleneck (compute-bound vs memory-bound)
```

### Parallel Patterns
- Five-step kernel optimisation methodology (check reduciton cuda codes)

- Step 1 — V1 baseline
- Step 2 — V2 fix divergence
- Step 3 — V3 better load
- Step 4 — V4 minimise global memory
- Step 5 — V5 thread coarsening

---

Contact: charanshankar629@gmail.com | Dublin, Ireland

Date : 29th March 2026

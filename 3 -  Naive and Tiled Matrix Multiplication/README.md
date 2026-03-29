# Day 3 — Memory Hierarchy and Tiled Matrix Multiplication

## The memory hierarchy

| Memory Type | Location | Latency | Size | Scope |
|-------------|----------|---------|------|-------|
| Registers | On-chip (SM) | ~1 cycle | 64KB/SM | Per thread |
| Shared memory | On-chip (SM) | ~5 cycles | 64KB/SM | Per block |
| Global memory | Off-chip (DRAM) | 400-800 cycles | 16GB (T4) | All threads |
| Constant memory | Off-chip + cached | ~5 cycles cached | 64KB | All threads (read-only) |

**The fundamental rule of GPU optimisation:**
Keep data in registers and shared memory. 
Minimise global memory accesses.


```
For each 16x16 tile:
  1. All 256 threads load their element into shared memory (1 global read each)
  2. __syncthreads() — wait for complete tile
  3. All 256 threads compute using shared memory (16 fast reads each)
  4. __syncthreads() — protect tile before next iteration
```

Each element of A is loaded from global memory once and reused 16 times.
Global memory reads reduced from 1024 to 32 per thread = 32x reduction.


## Files

| File | Description |
|------|-------------|
| `matmul.cu` | Naive vs tiled matrix multiplication with full comments |

## Results

```
Matrix size: 512 x 512
Tile size:   16 x 16
GPU:         NVIDIA T4

Naive matmul:  24.555 ms
Tiled matmul:   0.743 ms
Speedup:       47.92x
```

## Compile and run

```bash
nvcc -O2 -o matmul matmul.cu && ./matmul
```

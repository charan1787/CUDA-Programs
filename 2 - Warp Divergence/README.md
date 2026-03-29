# Day 2 — GPU Architecture: Warps, Divergence, and Occupancy


## Warp divergence
When threads in the same warp take different branches:
```
if (threadIdx.x % 2 == 0) {
    // Path A — even threads
} else {
    // Path B — odd threads
}
```
The warp cannot split. It executes Path A then Path B. 

Two serial passes instead of one.

The programs are run on Google Colab with T4 GPU : 

### T4 Specifications
```
Max warps per SM:      32
Max threads per SM:    1024
Max blocks per SM:     16
Shared memory per SM:  65536 bytes (64KB)
Registers per SM:      65536
Threads per warp:      32 (always)
```

### Occupancy calculation formula
```
Warps from threads:       floor(max_threads_SM / threads_per_block) × warps_per_block
Warps from registers:     floor(regs_SM / (regs_thread × threads_block)) × warps_per_block
Warps from shared mem:    floor(smem_SM / smem_block) × warps_per_block
Final occupancy = min(all three) / max_warps_SM
```

## Files

| File | Description |
|------|-------------|
| `divergence.cu` | Divergent vs non-divergent kernel benchmark |

## Results

```
Divergent kernel:     1.1752 ms
Non-divergent kernel: 0.0338 ms
Speedup:              34x
```

## Note : Why 34x difference?

Two compounding problems in the divergent kernel:

1. **Divergence** : Every warp splits 50/50 → both branches execute serially
2. **Strided access** : Divergent threads access every-other element → breaks coalescing

The 2nd Kernel eliminates both these problems

## compile and run

```bash
nvcc -O2 -o divergence divergence.cu && ./divergence
```


# Day 4 — Memory Coalescing and the Optimisation Checklist


When 32 threads in a warp read from global memory simultaneously,
the GPU memory controller checks whether the 32 addresses are consecutive.

**Coalesced access** — addresses are consecutive :
```
Thread 0 → address 0
Thread 1 → address 1
Thread 2 → address 2
...
Thread 31 → address 31
```
Memory controller: ONE transaction serves all 32 threads. Fast.

**Uncoalesced access** — addresses are scattered :
```
Thread 0 → address 0
Thread 1 → address 4096
Thread 2 → address 8192
...
Thread 31 → address 126976
```
Memory controller: 32 SEPARATE transactions needed. Slow.

## Why column access is uncoalesced :

A 4096×4096 matrix stored row-major in memory:
- Row 0: addresses 0 to 4095
- Row 1: addresses 4096 to 8191
- Row 2: addresses 8192 to 12287

Reading column 0 means reading in[0][0], in[1][0], in[2][0]...
These are at addresses 0, 4096, 8192... — each 4096 elements apart.

Each thread in the warp needs a separate cache line fetch = 32 transactions.

## The optimisation checklist : 

When you have a slow kernel, go through these in order :

```
1. Maximise occupancy
   — Too many registers per thread?
   — Too much shared memory per block?
   — Is block size appropriate?

2. Coalesce global memory accesses
   — Do consecutive threads access consecutive addresses?
   — If not — restructure the access pattern

3. Minimise global memory traffic
   — Can shared memory cache reused data?
   — Can constant memory serve read-only broadcast data?

4. Minimise control divergence
   — Do threads in the same warp take different branches?
   — Can you eliminate the branch mathematically?

5. Apply thread coarsening
   — Is each thread doing enough work?
   — Can you reduce redundant loads across blocks?

6. Know your bottleneck
   — Is the kernel memory-bound or compute-bound?
   — ALWAYS optimise the actual bottleneck — not the wrong one
```

## Files

| File | Description |
|------|-------------|
| `coalescing.cu` | Coalesced vs uncoalesced matrix copy benchmark |

## Results

```
Coalesced (row) average:    0.5762 ms
Uncoalesced (col) average:  1.2177 ms
Slowdown from uncoalescing: 2.11x
```

## Compile and run

```bash
nvcc -O2 -o coalescing coalescing.cu && ./coalescing
```


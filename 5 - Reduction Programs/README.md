# Day 5 — Parallel Reduction : Five-Step Optimisation

## What is reduction?

Reduction combines N elements into one value using an associative operation.

```
Sum reduction: [3, 1, 4, 1, 5, 9, 2, 6] → 31
Max reduction: [3, 1, 4, 1, 5, 9, 2, 6] → 9
```

**Sequential (CPU):** O(N) steps — loop through all elements.

**Parallel tree (GPU):** O(log N) steps — pair up elements, add pairs simultaneously.

## The five optimisation steps : 

### V1 — Naive (baseline with divergence)

### V2 — No Divergence

### V3 — Better Load (2 elements per thread)

### V4 — Minimise Global Memory (4 elements per thread)
Result: 12.48x speedup — hits the T4 memory bandwidth ceiling

### V5 — Thread Coarsening (8 elements per thread)


## Files

| File | Description |
|------|-------------|
| `reduction.cu` | All 5 versions in one benchmark program |
| `speedup_chart.png` | Visual comparison of all 5 versions |

## Results

```
Version         Time (ms)   Speedup   What was fixed
─────────────────────────────────────────────────────
V1 Naive        2.9528      1.00x     Baseline
V2 No diverge   1.0403      2.84x     Warp divergence
V3 Better load  0.3773      7.83x     Thread underutilisation
V4 Min global   0.2365      12.48x    Global memory traffic
V5 Coarsened    0.2441      12.10x    (Slower — bandwidth saturated in V4 itself)



## Compile and run

```bash
nvcc -O2 -o reduction reduction.cu && ./reduction
```


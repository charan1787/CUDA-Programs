# 1 — Thread Hierarchy and Indexing 

## The thread hierarchy
CUDA organises threads in a three-level hierarchy:

```
Grid
├── Block 0  (threads 0 to blockDim.x-1)
├── Block 1  (threads 0 to blockDim.x-1)
├── Block 2  (threads 0 to blockDim.x-1)
└── ...
```

Every thread has a unique global ID computed as:
```
i = blockIdx.x * blockDim.x + threadIdx.x
```

Think of it like apartment addresses:
- blockIdx.x = floor number
- blockDim.x = apartments per floor
- threadIdx.x = apartment number on that floor

### Why bounds checking is essential
We always launch more threads than we have data elements because
block sizes must be fixed powers of 2. The extra threads must be
prevented from writing to memory that does not belong to our array.

```cpp
if (i < N) {
    C[i] = A[i] + B[i];  // Only valid threads do work
}
```

### 2D indexing for matrices
For matrix operations, we use 2D grids and blocks:
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * N + col;  // Row-major flat index
```

### Ceiling division for grid size
To cover N elements with blocks of size B:
```cpp
int numBlocks = (N + B - 1) / B;  // Always rounds up
```
For N=1000, B=256: (1000+255)/256 = 4 blocks (covers 1024 threads)

## Files

| File | Description |
|------|-------------|
| `vec_add.cu` | 1D vector addition — first CUDA kernel |
| `matrix_add.cu` | 2D matrix addition — 2D indexing |

## Results

| Program | N | Time |
|---------|---|------|
| vec_add | 1000 | ~0.187 ms |
| matrix_add | 1000x1000 | ~0.X ms |

## compile and run

```bash
nvcc -O2 -o vec_add vec_add.cu && ./vec_add
nvcc -O2 -o matrix_add matrix_add.cu && ./matrix_add
```
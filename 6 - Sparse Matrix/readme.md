# Sparse Matrix Vector Multiplication (SpMV) with CSR Format

Implemented Sparse Matrix Vector Multiplication (SpMV) using the
CSR (Compressed Sparse Row) format on GPU.

Computes y = A * x where:
- A is a sparse matrix stored in CSR format
- x is a dense input vector
- y is the dense output vector

## Why sparse matrices matter

In deep learning, many matrices are mostly zeros:
- Pruned neural networks — weights below threshold set to zero
- Graph neural networks — most nodes not connected to each other
- Sparse attention — tokens only attend to a few other tokens

Storing and computing with all those zeros wastes memory and compute.

Sparse formats store only the non-zero values and skip the zeros entirely.

---

## The three sparse matrix formats

### COO — Coordinate Format

Simplest format. Store row, column, and value for every non-zero.

row    = [0, 0, 1, 2, 3]
col    = [0, 3, 1, 2, 3]
values = [1, 2, 3, 4, 5]

Good for : building the matrix

Bad for : GPU computation — no structure to exploit

### CSR — Compressed Sparse Row

Groups non-zeros by row. Stores where each row starts.

Three arrays :
- values[]  — all non-zero values, row by row
- colIdx[]  — column index of each non-zero
- rowPtr[]  — where each row starts in values[]

Good for : GPU SpMV — one thread per row, parallel across all rows

Bad for : irregular row lengths cause load imbalance

### ELL — ELLPACK Format

Pads all rows to the same length — maximum non-zeros per row.

Every thread does the same amount of work. Perfect load balance.

Good for : GPU SpMV when rows have similar non-zero counts

Bad for : irregular sparsity — one long row wastes memory for all rows

## CSR format : 

Our sparse matrix A (4x4) :

1 0 0 2

0 3 0 0

0 0 4 0

0 0 0 5

5 non-zeros out of 16 elements — 69% zeros.

### CSR representation :

values = [1, 2, 3, 4, 5]

colIdx = [0, 3, 1, 2, 3]

rowPtr = [0, 2, 3, 4, 5]

### How to read rowPtr :

Row 0: rowPtr[0]=0 to rowPtr[1]=2 -> values[0,1] = {1,2} at cols {0,3}

Row 1: rowPtr[1]=2 to rowPtr[2]=3 -> values[2]   = {3}   at col  {1}

Row 2: rowPtr[2]=3 to rowPtr[3]=4 -> values[3]   = {4}   at col  {2}

Row 3: rowPtr[3]=4 to rowPtr[4]=5 -> values[4]   = {5}   at col  {3}

## Where SpMV is used in deep learning

| Application | Why sparse |
|-------------|-----------|
| Pruned networks | 90% of weights zeroed after pruning |
| Graph neural networks | Adjacency matrix — most nodes unconnected |
| Sparse attention | Each token attends to only a few others |
| Recommendation systems | Most user-item pairs unrated |

In all these cases, SpMV replaces dense GEMM.

cuSPARSE provides production-optimised SpMV for these workloads.



## Google Colab:
```python
%%writefile spmv_csr.cu
# paste code here

!nvcc -O2 -o spmv_csr spmv_csr.cu && ./spmv_csr
```
Date : 02 April 2026
Email : charanshankar629@gmail.com

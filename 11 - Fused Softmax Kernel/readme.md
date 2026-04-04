# Fused Softmax CUDA Code 

Kernel 1: find max value       		→ write result to global memory
Kernel 2: subtract max + exp   	    → write result to global memory
Kernel 3: find sum             		→ write result to global memory
Kernel 4: divide by sum        		→ write result to global memory

This is called memory-bound overhead. The kernels themselves are fast. 

The bottleneck is the constant reading and writing to slow global memory between them.

A fused kernel does all four operations in one single kernel launch, keeping intermediate results in fast registers and shared memory instead of writing them to global memory.

One read. One write. Everything in between happens in fast on-chip memory.

This is the core idea behind every fused operation in deep learning : 
reduce the number of times data travels between the GPU chip and DRAM.

```bash
Real PyTorch unfused:
  5 kernel launches
  5 global memory reads + 5 global memory writes = 10 accesses
  Launch overhead × 5

Our unfused simulation:
  1 kernel launch
  ~7 global memory accesses
  Simulates the memory cost but not the launch overhead

Our fused kernel:
  1 kernel launch
  ~3 global memory accesses
  Minimum possible memory traffic
```
## OutPut Resuklts : 

```bash
Unfused softmax: 1.5045 ms
Fused softmax:   0.1368 ms
Speedup:         10.99x

First block sum (unfused): 1.000000 
First block sum (fused):   1.000000 
```

### Inbuilt softmax runs with fused mode inside. But when we run custum functions or commands of the softmax seperatly we end up launching 5 different kerneles which increases the global memory access. 

## How to compile and run : 

```python
Google colab with T4 GPU: 

%%writefile fused_softmax.cu
// copy the code here

Run : !nvcc -O2 -o fused_softmax fused_softmax.cu && ./fused_softmax
```

Date : 04/04/2026

Email : charanshankar629@gmail.com
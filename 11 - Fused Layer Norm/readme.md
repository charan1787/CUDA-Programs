# Fused Layer Normalisation

## What this program does
Implemented and benchmarks unfused vs fused layer normalisation on GPU.
Showed how keeping mean and variance in registers/shared memory
instead of global memory gives 10.51x speedup.

## Layer norm equation
  mean     = sum(x) / N
  variance = sum((x - mean)^2) / N
  output   = gamma * (x - mean) / sqrt(variance + epsilon) + beta

## What is layer normalisation?
Solves internal covariate shift — values after many matrix
multiplications can become very large or very small, making
training unstable. Layer norm normalises each layer's output
to mean=0, std=1, then applies scale (gamma) and shift (beta) 
so the network can undo normalisation if needed.

## Why fusion helps
Unfused — 3 passes, mean and variance written to global memory :
Fused — 1 kernel, mean and variance stay in registers :

Key insight: sdata is reused for variance because once mean
is extracted into a register, the shared memory is free to 
be overwritten.

## transformer application
Appears 2 times per transformer block :
  Input -> Layer Norm 1 -> Attention -> Add -> Layer Norm 2 -> FFN -> Add

GPT-3 (96 layers) forward pass calculation :
  96 layers × 2 norm per block × 32 batch × 2048 tokens
  = 12,582,912 layer norm calls per forward pass

## Compile and run

Google Colab T4 : 

```python
%%writefile fused_layer_norm.cu
// paste code here 
nvcc -O2 -o fused_layernorm fused_layernorm.cu && ./fused_layernorm
```

---
Date : 05 April 2026

Email : charanshankar629@gmail.com

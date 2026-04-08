# Simplified FlashAttention

Implemented naive attention and FlashAttention style tiled attention.
Demonstrated the core IO-aware tiling principle that makes long-context LLMs practical.

## The attention formula
  Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V

  Q = what each token is looking for
  K = what each token contains
  V = what each token gives to others

## The core memory problem with naive attention
For sequence length N, naive attention stores the full
[N x N] score matrix in global memory — O(N^2) memory.

  N=1024:  4MB per attention head
  N=4096:  64MB per attention head
  N=16384: 1GB per attention head — does not fit on GPU

For large N, scores[] spills from registers to DRAM,
dramatically increasing latency.

## FlashAttention is the solution

1. Never materialise [N x N] matrix in global memory
   Process Q, K, V in tiles that fit in registers
   Memory usage : O(N) instead of O(N^2)

2. Fusing three operations into one kernel
   QK^T computation + softmax + weighted V sum
   No intermediate results written to global memory

3. Online softmax eliminates a complete pass
   Maintain running max and sum across tiles
   Correct accumulated values when max changes :

     correction = exp(old_max - new_max)
     running_sum *= correction
     acc[]       *= correction
   Each tile processed exactly once

## Online softmax — the mathematical key
Standard softmax needs to see all scores before computing.
Online softmax maintains running statistics :
  running_max : maximum score seen so far
  running_sum : normalisation denominator
  acc[]       : accumulated weighted V sum

## Results : NVIDIA T4 in Google Colab
  N=64, d=8
  Runs: 100-iteration average

  Naive attention:  7.1014 ms
  Flash attention:   0.1492 ms
  Speedup:          47.60x

## Memory comparison
  Naive :  O(N^2) = 64 x 64 = 4096 floats = 16KB (our example)
  Flash :  O(N)   = tile registers only

  For N = 4096 (for LLMs):
  Naive :  4096 x 4096 x 4 = 64MB per attention head
  Flash :  tile size only will scale to 128k+ context (GPT latest and CLaude)

## Why this matters
Claude's 200k context, GPT-4's 128k context — only possible
because FlashAttention eliminates the O(N^2) memory requirement.
Without it these models would exceed GPU memory at inference time.

## How to compile and run
  nvcc -O2 -o flash_attention flash_attention.cu -lm && ./flash_attention

Date : 8th April 2026 <br/>
Email : charanshankar629@gmail.com

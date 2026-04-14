# GPT-2 Profiling with torch.profiler


Profiled GPT-2 (124M params) forward pass on NVIDIA T4
using torch.profiler. Identified which operations consume most
GPU time and measured how time scales with sequence length.

## Model
GPT-2 small: 12 layers, 12 heads, d_model=768, d_ff=3072
Parameters: 124.4M
Size: 475MB in FP32

## Profiling results (batch=8, seq_len=128)
  Forward pass average: 57.33ms
  Throughput: 17,862 tokens/second

  Top operations by GPU time:
    aten::addmm (GEMM):     73.54% — Linear layers dominate
    Elementwise (mul, add): 15.30% — Residual connections
    FlashAttention:          4.07% — Small at short sequences
    GELU (tanh + pow):       4.02% — Two separate kernels

## Sequence length scaling
  64  -> 128: 1.85x  (linear)
  128 -> 256: 2.06x  (linear)
  256 -> 512: 1.98x  (linear)
  Confirms O(N) FFN dominance — FFN scales linearly with tokens

## Key insights from the results : 
1. GEMM dominates at short sequences — FFN compute-bound
2. Attention only 4% at seq_len=128 — O(N²) not yet significant
3. GELU uses two separate kernels (tanh + pow) — fusion opportunity identified for optimisation next


April 12th 2026<br\>
charanshankar629@gmail.com

# GPT-2 Profiling with torch.profiler


Profiled GPT-2 (124M params) forward pass on NVIDIA T4
using torch.profiler. Identified which operations consume most
GPU time and measured how time scales with sequence length.

## Model
GPT-2 small: 12 layers, 12 heads, d_model=768, d_ff=3072<br\>
Parameters: 124.4M<br\>
Size: 475MB in FP32<br\>

## Profiling results (batch=8, seq_len=128)
  Forward pass average: 57.33ms<br\>
  Throughput: 17,862 tokens/second<br\>

  Top operations by GPU time:<br\>
    aten::addmm (GEMM):     73.54% — Linear layers dominate<br\>
    Elementwise (mul, add): 15.30% — Residual connections<br\>
    FlashAttention:          4.07% — Small at short sequences<br\>
    GELU (tanh + pow):       4.02% — Two separate kernels<br\>

## Sequence length scaling
  64  -> 128: 1.85x  (linear)<br\>
  128 -> 256: 2.06x  (linear)<br\>
  256 -> 512: 1.98x  (linear)<br\>
  Confirms O(N) FFN dominance — FFN scales linearly with tokens<br\>

## Key insights from the results : 
1. GEMM dominates at short sequences — FFN compute-bound<br\>
2. Attention only 4% at seq_len=128 — O(N²) not yet significant<br\>
3. GELU uses two separate kernels (tanh + pow) — fusion opportunity identified for optimisation next<br\>


April 12th 2026<br\>
charanshankar629@gmail.com

# Fused Attention with Causal Masking

## What this program does
Extended FlashAttention with causal masking.
Implemented full attention and causal attention and benchmarked both outputs.


## What is causal masking?
Prevents each token from attending to future tokens.
Sets upper triangle of [N x N] attention matrix to -infinity.
After softmax it will get zero attention to future tokens.

Attention matrix (4 tokens):<br/>
tok0  tok1  tok2  tok3<br/>
tok0 [ *    -INF    -INF    -INF  ]  attends to 1 token<br/>
tok1 [ *     *      -INF    -INF  ]  attends to 2 tokens<br/>
tok2 [ *     *       *      -INF  ]  attends to 3 tokens<br/>
tok3 [ *     *       *         *  ]  attends to all 4<br/>

## Why GPT needs causal masking ?
GPT generates text left to right one token at a time.
During training, the model sees the full sequence but must
learn to predict each token using only past context.

Without masking : token 5 could attend to tokens 6,7,8
copying the answer from future tokens instead of learning.
Model would train perfectly but fail completely at predicting
because future tokens do not exist during generation.

## Why causal attention is faster
Processes only lower triangle — roughly N^2/2 computation.

Total work ~N^2/2 vs N^2 for full attention.

Combined with FlashAttention tiling: 73x measured speedup.

## Results on NVIDIA T4 in colab
  N=64, d=8, runs=100 average

  Full attention:            15.73 ms
  Causal attention:           0.21 ms
  Causal speedup:            73.30x

## How to compile and run
  nvcc -O2 -o causal_attention causal_attention.cu -lm && ./causal_attention

Date : 8th April 2026 <br/>
Email : charanshankar629@gmail.com

# INT8 Per-Channel Quantisation Kernel

Implemented per-channel INT8 quantisation and dequantisation
on GPU. Converted FP32 weights to INT8 for 4x memory reduction
with less than 0.4% accuracy loss.

## The quantisation formula
  scale     = max(|weights_in_channel|) / 127
  int8_val  = round(fp32_val / scale)<br/>
  fp32_approx = int8_val * scale<br/>

## Per-channel vs per-tensor
Per-channel : one scale per output channel
  Each channel optimally scaled for its own weight range<br/>
  Better accuracy for production standard

Per-tensor : one scale for entire matrix
  Fails when channels have different weight ranges<br/>
  Small weights in low-range channels round to zero

## Thread organisation
  One block per channel — 64 blocks for 64 channels
  256 threads per block<br/>
  All blocks run simultaneously<br/>

  Within each block:
    Step 1: Each thread finds local max of its elements<br/>
    Step 2: Tree reduction → global max for channel<br/>
    Step 3: scale = max_abs / 127<br/>
    Step 4: Each thread quantises its elements

## Results on NVIDIA T4 (Google colab)
  Matrix: 64 channels x 256 features<br/>
  FP32 size: 64KB → INT8 size: 16KB

  Quantisation time:   105.7711 ms (one-time offline)<br/>
  Dequantisation time: 0.0287 ms   (every forward pass)

  Max error:      0.3936% of max weight value<br/>
  Mean error:     0.1940%<br/>
  Compression:    3.94x (scale factor overhead explains gap from 4x)

## Why loading speedup is small on this test :
  64KB matrix fits entirely in T4 L2 cache (4MB)<br/>
  Both FP32 and INT8 served at cache speed — no bandwidth difference<br/>
  Real LLMs (7B+ params) exceed cache and hit DRAM bandwidth<br/>
  At DRAM : INT8 moves half the data so 2x throughput improvement

## Why quantisation is slow but dequantisation is fast
  Quantisation : max reduction (O(log N) steps) + quantise pass<br/>
  Dequantisation : one multiply per element and no reduction needed<br/>
  In production : quantise once offline, dequantise in every forward pass<br/>

## How to compile and run
  nvcc -O2 -o quantisation_int8 quantisation_int8.cu -lm && ./quantisation_int8

Date : 12th April 2026 <br/>
Email : charanshankar629@gmail.com
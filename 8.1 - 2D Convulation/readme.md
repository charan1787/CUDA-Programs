# 2D Convolution with Constant Memory

Implemented 2D convolution on GPU with the filter stored in
constant memory. Demonstrated edge detection and Gaussian smoothing.

## What is 2D convolution?
Slides a small filter over an input image. At each position,
computes the dot product between the filter and the image patch.
One GPU thread per output pixel. All pixels computed simultaneously.


Every conv layer in ResNet, VGG, EfficientNet uses this operation.
cuDNN implements it faster using im2col — reshaping input patches
into a matrix then applying GEMM (same as FC layer from Program 2).

Convolution -> im2col -> matrix multiply -> same as tiled matmul

## How to compile and run
```bash
nvcc -O2 -o conv2d conv2d.cu && ./conv2d
```

---
Date : 03 April 2026

Email : charanshankar629@gmail.com

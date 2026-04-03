# CUDA GPU : Fully Connected Layer Forward Pass + ReLU

Implemented a fully connected (FC) layer forward pass followed by
ReLU activation entirely in CUDA.

Computes :
  output = input × weights + bias   (FC layer)
  output = max(0, output)           (ReLU activation)

This is exactly what PyTorch's nn.Linear + nn.ReLU does under the hood, a GEMM operation via cuBLAS followed by an element-wise activation kernel.

A FC layer connects every input neuron to every output neuron.
For each output neuron, it computes a weighted sum of all inputs
plus a bias term.

### Text Book Example : 
Think of it like a doctor diagnosing diseases:
- 128 symptoms (input features) per patient
- 64 possible diseases (output neurons)
- Each disease has its own set of 128 weights
  (how important each symptom is for that disease)
- Bias is the baseline probability regardless of symptoms

input:   [64 × 128]   (64 images, 128 features each)

weights: [128 × 64]   (128 inputs → 64 outputs)

output:  [64 × 64]    (64 images, 64 outputs each)

output = input × weights + bias.

-> This is GEMM — General Matrix Multiply.
-> Identical to the tiled matrix multiplication.
-> cuBLAS implements this with tensor cores for production use.

## ReLU activation

Positive values are unchanged.
Negative values become zero.

### GPU implementation

The simplest possible CUDA kernel:
- One thread per element
- No shared memory needed
- No synchronisation needed
- Pure element-wise operation
```cpp
data[i] = fmaxf(0.0f, data[i]);
```

## The complete forward pass pipeline
Input [64 × 128] : 

FC Layer (GEMM) :
output = input × weights + bias
64 × 128 matrix × 128 × 64 matrix
= 64 × 64 output matrix

ReLU :
output = max(0, output)
Element wise - one thread per element

Output : [64 × 64]

### Google Colab:
```python
%%writefile fc_forward.cu
# paste code here

!nvcc -O2 -o fc_forward fc_forward.cu && ./fc_forward
```

Date : 02 April 2026
Email : charanshankar629@gmail.com

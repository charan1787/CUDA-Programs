# GPU CUDA : Softmax

Implemented numerically stable softmax on GPU using CUDA.
Converts raw scores into probabilities that sum to 1.0.

Used in:
- Final layer of classification networks
- Attention mechanism in transformers (scaled dot-product attention)
- Any operation requiring a probability distribution over classes

## The softmax equation

For a vector z of length N, softmax at position i:

softmax(z)[i] = exp(z[i]) / sum(exp(z[j]) for all j)

### Step by step example

Input logits : [2.0, 1.0, 0.5, -1.0]

Step 1 — Exponentiate each element :

exp(2.0)  = 7.389
exp(1.0)  = 2.718
exp(0.5)  = 1.649
exp(-1.0) = 0.368

Step 2 — Sum all exponentials :

sum = 7.389 + 2.718 + 1.649 + 0.368 = 12.124

Step 3 — Divide each by sum :

softmax[0] = 7.389 / 12.124 = 0.609
softmax[1] = 2.718 / 12.124 = 0.224
softmax[2] = 1.649 / 12.124 = 0.136
softmax[3] = 0.368 / 12.124 = 0.030

Sum = 0.609 + 0.224 + 0.136 + 0.030 = 1.000 

## The numerical stability problem

### What goes wrong without the fix

If any logit is very large like 1000 : 

exp(1000) = Inf   (overflows float representation)
Inf / Inf    = NaN (not a number)

The entire computation silently produces NaN.
Your network outputs garbage. Training diverges.
This is called numerical overflow.

### The fix — subtract the maximum

Numerically stable softmax :

max_val = max(z)
softmax(z)[i] = exp(z[i] - max_val) / sum(exp(z[j] - max_val))

Subtracting max makes the largest value exp(0) = 1.0.
All other values are smaller. No overflow possible.

Identical result — no overflow.

## How softmax maps to CUDA kernels

Softmax requires five operations :

Step 1 : max_val = max(z)          → MAX REDUCTION  
Step 2 : z[i] = z[i] - max_val    → element wise subtraction
Step 3 : z[i] = exp(z[i])         → element-wise exp
Step 4 : sum_val = sum(z)          → SUM REDUCTION   
Step 5 : z[i] = z[i] / sum_val    → elementwise division

## How to compile and run

Google Colab:
```python
%%writefile softmax.cu
# paste code here

!nvcc -O2 -o softmax softmax.cu && ./softmax
```
Date : 03 April 2026
Email : charanshankar629@gmail.com




# Custom Kernel extension to PyTorch 

A Python-callable GPU function that computes the sum of a PyTorch tensor using your optimised reduction kernel.

### Use Case
x = torch.ones(1 << 24, device='cuda')
result = my_reduction.forward(x)
print(result)

Under the hood custom CUDA kernel runs. Not inbuilt torch.sum(). 
-----------------------------
PyTorch provides a mechanism called torch.utils.cpp_extension that lets you compile C++ and CUDA code at runtime and import it as a Python module.

## File Explanation : 

reduction_kernel.cu  — the actual GPU computation
                       same V4 kernel from Day 5
                       wrapped to accept PyTorch tensors

wrapper.cpp          — the bridge between Python and CUDA
                       pybind11 exposes forward() to Python
                       one line per function you want to expose

setup.py             — tells PyTorch how to compile everything
                       lists source files, sets extension name

## OutPut : 

Correctness: 16777216 == 2^24    

Benchmark:
Size    Custom(ms)    torch.sum(ms)    Ratio
1M      0.0431        0.0229           1.88x slower
4M      0.1298        0.0742           1.75x slower
16M     0.4692        0.2622           1.79x slower
64M     1.3230        0.9740           1.36x slower

### Simple Analogy : 

reduction_kernel.cu contains the GPU kernel and the C++ function that bridges PyTorch tensors to raw CUDA pointers. 

wrapper.cpp use pybind11 to expose that C++ function to Python so it can be called as my_reduction.forward(). 

setup.py tells PyTorch's build system how to compile both files together into one importable Python module.

### data_ptr<float>()

It extracts the raw GPU memory pointer from a high-level PyTorch tensor. The CUDA kernel only understands raw float pointers — it has no knowledge of PyTorch objects. This function bridges the two worlds."

## Compile and Run : 

- Load these 3 files in the Google Colab.
- Run Compile.py
- Run test.py (to test the code)

Date : 03 April 2026
Email : charanshankar629@gmail.com



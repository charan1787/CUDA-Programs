#include <torch/extension.h>

torch::Tensor reduction_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &reduction_forward, "Custom CUDA reduction");
}

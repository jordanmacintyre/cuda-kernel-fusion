# CUDA Kernel Fusion

Educational repository demonstrating **CUDA kernel fusion** techniques to optimize GPU performance by reducing memory bandwidth bottlenecks. Includes fused kernel implementations achieving **2.3-7.1x speedup** over PyTorch by keeping intermediate values in GPU registers instead of global memory.

## What is Kernel Fusion?

**Problem:** Each PyTorch operation launches a separate kernel with expensive memory round-trips:

```python
# PyTorch - 3 kernels, 7 memory operations
a = x + y          # Kernel 1: read x,y → write a
b = a * 2          # Kernel 2: read a → write b
c = torch.exp(b)   # Kernel 3: read b → write c
```

**Solution:** Single fused kernel keeps intermediates in registers:

```cuda
// Fused CUDA - 1 kernel, 3 memory operations
float a = x[idx] + y[idx];   // register (fast!)
float b = a * 2.0f;          // register (fast!)
output[idx] = __expf(b);     // write to memory
```

**Result:** 2.3x speedup, 98% efficiency, < 1e-6 relative error

## Quick Start

```bash
# Setup
conda create -n cuda-fusion python=3.12
conda activate cuda-fusion
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c nvidia cuda-toolkit ninja
pip install pytest tqdm numpy

# Install package
pip install --no-build-isolation -e .

# Run tests and benchmarks
pytest tests/ -v
python benchmarks/bench_add_mul_exp.py
```

## Implemented Kernels

### 1. Element-wise Fusion: `add_mul_exp`
Fuses `exp((x + y) * 2)` into a single kernel.
- **PyTorch**: 3 kernels, 7 memory ops
- **CUDA**: 1 kernel, 3 memory ops
- **Speedup**: 2.32x (98% efficiency)

### 2. INT8 Quantization: `quantize_int8`
Fuses `clamp(round(x / scale + zero_point), -128, 127).to(int8)` into a single kernel.
- **PyTorch**: 5 kernels, 10 memory ops
- **CUDA**: 1 kernel, 2 memory ops
- **Speedup**: 7.14x (100% efficiency + 1.51x cache benefit)

## Usage

```python
import torch
from ops.cuda import add_mul_exp_cuda, quantize_int8_cuda
from ops.torch import add_mul_exp_pytorch, quantize_int8_pytorch

# Element-wise fusion
x = torch.randn(1_000_000, device='cuda')
y = torch.randn(1_000_000, device='cuda')
result = add_mul_exp_cuda(x, y)  # CUDA fused
baseline = add_mul_exp_pytorch(x, y)  # PyTorch unfused

# INT8 quantization
x = torch.randn(1_000_000, device='cuda')
quantized = quantize_int8_cuda(x, scale=4.2, zero_point=1.2)
```

**Naming Convention:**
- `operation_cuda()` - CUDA implementation (fused)
- `operation_pytorch()` - PyTorch baseline (unfused)

## Performance

Benchmarks on **RTX 3070** with **10M elements**:

### add_mul_exp: `exp((x + y) * 2)`
```
PyTorch:  0.690ms  (3 kernels, 7 memory ops, 267 MB traffic)
CUDA:     0.297ms  (1 kernel,  3 memory ops, 114 MB traffic)
Speedup:  2.32x    (98.0% roofline efficiency, max rel error: 8.9e-07)
```

### quantize_int8: FP32 → INT8 conversion
```
PyTorch:  0.911ms  (5 kernels, 10 memory ops, 381 MB traffic)
CUDA:     0.128ms  (1 kernel,   2 memory ops,  76 MB traffic)
Speedup:  7.14x    (100% roofline efficiency + 1.51x cache benefit, exact match)
```

**Why is it faster?**

Modern GPUs are **memory-bound**. Memory access is ~400x slower than register operations. Kernel fusion eliminates intermediate memory round-trips by keeping values in registers.

**Performance metrics explained:**
- **Roofline efficiency**: How close the kernel is to theoretical peak performance (100% = perfect)
- **Cache benefit**: When actual performance exceeds DRAM-only predictions due to L2 cache hits (>1.0x is good!)
- The quantize_int8 kernel achieves 100% efficiency AND benefits from cache, making it 1.51x faster than theory predicts

## Project Structure

```
cuda-kernel-fusion/
├── ops/                           # Kernel implementations
│   ├── cuda/                     # CUDA fused kernels (JIT compiled)
│   │   ├── add_mul_exp.py       # Element-wise fusion wrapper
│   │   ├── quantize_int8.py     # INT8 quantization wrapper
│   │   └── csrc/                # CUDA kernel source files
│   │       ├── add_mul_exp.cu   # Element-wise fusion kernel
│   │       └── quantize_int8.cu # Quantization kernel
│   └── torch/                    # PyTorch baseline implementations
│       ├── add_mul_exp.py       # Unfused PyTorch ops
│       └── quantize_int8.py     # Unfused quantization
├── tests/                         # Comprehensive test suite
│   ├── test_add_mul_exp_cuda.py      # CUDA tests
│   ├── test_add_mul_exp_pytorch.py   # PyTorch tests
│   ├── test_quantize_int8_cuda.py    # INT8 CUDA tests
│   └── test_quantize_int8_pytorch.py # INT8 PyTorch tests
├── benchmarks/                    # Performance analysis framework
│   ├── utils/                    # Reusable benchmarking utilities
│   │   ├── performance.py       # Timing & profiling tools
│   │   └── analysis.py          # Numerical accuracy analysis
│   ├── bench_add_mul_exp.py      # Element-wise fusion benchmark
│   ├── bench_quantize_int8.py    # Quantization benchmark
│   └── bench_template.py         # Template for new benchmarks
├── pyproject.toml                # Package config & dependencies
└── README.md
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific kernel tests
pytest tests/test_add_mul_exp_cuda.py -v
pytest tests/test_quantize_int8_cuda.py -v
```

**Test coverage:**
- Correctness (basic ops, known values, various sizes)
- Input validation (CPU/GPU tensors, dtype checks)
- Edge cases (zeros, large values, numerical stability)
- Numerical accuracy (relative error < 1e-5 for FP32)

## Benchmarking

```bash
# Run benchmarks for each kernel
python benchmarks/bench_add_mul_exp.py
python benchmarks/bench_quantize_int8.py
```

**Each benchmark provides:**
- Detailed timing statistics (warmup, mean, median, std)
- Numerical accuracy analysis (abs/rel errors, percentiles)
- Memory traffic analysis (ops count, bandwidth, efficiency)
- PyTorch profiler output (kernel-level timing)

See [benchmarks/README.md](benchmarks/README.md) for the benchmarking framework documentation.

## Adding New Operations

### 1. Create CUDA Kernel

`ops/cuda/csrc/your_op.cu`:
```cuda
#include <torch/extension.h>

__global__ void your_op_kernel(const float* x, const float* y,
                                float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Your fused operations here
        float temp = x[idx] + y[idx];
        output[idx] = temp * 2.0f;
    }
}

torch::Tensor your_op_cuda(torch::Tensor x, torch::Tensor y) {
    auto output = torch::empty_like(x);
    int threads = 256;
    int blocks = (x.numel() + threads - 1) / threads;

    your_op_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(),
        output.data_ptr<float>(), x.numel()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("your_op_cuda", &your_op_cuda);
}
```

### 2. Create Python Wrapper and Baseline

`ops/cuda/your_op.py`:
```python
import torch
from torch.utils.cpp_extension import load

# Compile CUDA extension
_C = load(name="your_op_cuda", sources=["csrc/your_op.cu"], ...)

def your_op_cuda(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CUDA implementation."""
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")
    return _C.your_op(x, y)

def your_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Main API - dispatches to CUDA."""
    return your_op_cuda(x, y)
```

`ops/torch/your_op.py`:
```python
import torch

def your_op_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """PyTorch baseline implementation."""
    # Your unfused PyTorch code here
    temp = x + y
    return temp * 2
```

### 3. Add Tests and Benchmarks

```bash
# Copy templates
cp tests/test_add_mul_exp.py tests/test_your_op.py
cp benchmarks/bench_template.py benchmarks/bench_your_op.py

# Edit and run
pytest tests/test_your_op.py -v
python benchmarks/bench_your_op.py
```

## Key CUDA Concepts

**Thread Indexing:**
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

**Kernel Launch:**
```cuda
int threads = 256;  // Threads per block
int blocks = (size + threads - 1) / threads;  // Ceiling division
my_kernel<<<blocks, threads>>>(args);
```

**Memory Hierarchy (latency):**
- Registers: 1 cycle ← Keep data here!
- Shared memory: ~20 cycles
- Global memory: ~400 cycles ← Avoid round-trips

## Operations That Benefit from Fusion

1. **Activation functions:** `sigmoid(x * w + b)`, `gelu(x)`, `swish(x)`
2. **Normalization:** `layer_norm(x)`, `rms_norm(x)`
3. **Element-wise chains:** `(x + y) * z + w`, `dropout(relu(x))`

## Troubleshooting

| Error | Solution |
|-------|----------|
| `nvcc not found` | `conda install -c nvidia cuda-toolkit` |
| `ninja not found` | `conda install ninja` |
| Slow compilation | Set `export TORCH_CUDA_ARCH_LIST='8.6'` |
| Import error | Run `pip install --no-build-isolation -e .` |
| Tests fail | Ensure `torch.cuda.is_available()` returns True |

## Resources

- 📖 [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- 🔧 [PyTorch Custom Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- 💡 [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- 📊 [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)

## License

MIT - See LICENSE file

---

**Author:** Jordan MacIntyre
**Purpose:** Educational demonstration of CUDA kernel fusion techniques
**GPU Tested:** RTX 3070 (compute capability 8.6)

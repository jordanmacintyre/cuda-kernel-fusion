# CUDA Kernel Fusion

Educational repository demonstrating **CUDA kernel fusion** techniques to optimize GPU performance by reducing memory bandwidth bottlenecks. Includes fused kernel implementations achieving **4.4-7.5x speedup over raw PyTorch** and **1.2-2.4x speedup over torch.compile** by keeping intermediate values in GPU registers instead of global memory.

## What is Kernel Fusion?

**Problem:** Each PyTorch operation launches a separate kernel with expensive memory round-trips:

```python
# PyTorch - 5 kernels, 10 memory operations
# quantize_int8: clamp(round(x / scale + zero_point), -128, 127).to(int8)
a = x / scale            # Kernel 1: read x, scale → write a
b = a + zero_point       # Kernel 2: read a, zero_point → write b
c = torch.round(b)       # Kernel 3: read b → write c
d = torch.clamp(c, -128, 127)  # Kernel 4: read c → write d
output = d.to(torch.int8)      # Kernel 5: read d → write output
```

**Solution:** Single fused kernel keeps intermediates in registers:

```cuda
// Fused CUDA - 1 kernel, 2 memory operations (read + write)
float val = input[idx];                      // read from VRAM
float scaled = val * inv_scale + zero_point; // FMA in registers (fast!)
int quantized = __float2int_rn(scaled);      // round in registers (fast!)
quantized = max(-128, min(127, quantized));  // clamp in registers (fast!)
output[idx] = (int8_t)quantized;             // write to VRAM
```

**Result:** 7.5x speedup over raw PyTorch (50M elements), 1.05x over torch.compile, exact integer matching

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

### 1. INT8 Quantization: `quantize_int8`
Fuses `clamp(round(x / scale + zero_point), -128, 127).to(int8)` into a single kernel.
- **PyTorch**: 5 separate kernels, 10 memory operations
- **Custom CUDA**: 1 fused kernel, 2 memory operations
- **Speedup vs raw PyTorch**: 7.5x (50M elements), 4.4x (500K elements)
- **Speedup vs torch.compile**: 2.4x (1M elements), 1.05x (50M elements)

### 2. Element-wise Fusion: `add_mul_exp`
Fuses `exp((x + y) * 2)` into a single kernel.
- **PyTorch**: 3 separate kernels, 7 memory operations
- **Custom CUDA**: 1 fused kernel, 3 memory operations
- **Speedup vs raw PyTorch**: 2.37x (100K elements), 2.32x (20M elements)
- **Speedup vs torch.compile**: 3.83x (100K elements), 1.06x (20M elements)

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

Benchmarks compare **PyTorch**, **torch.compile**, and **Custom CUDA** across multiple data sizes to show cache effects and memory bandwidth limits.

### quantize_int8: FP32 → INT8 conversion

**Test scenarios:**
- **L2 Cache**: 500K-1M elements (2-4 MB) - Fits in GPU L2 cache for maximum performance
- **VRAM**: 5M-50M elements (20-200 MB) - Exceeds cache, tests memory bandwidth limits

```
RAW EXECUTION TIME
Memory Location    Data Size      # Elements            PyTorch      torch.compile        Custom CUDA
                        (MB)                               (ms)               (ms)               (ms)
--------------------------------------------------------------------------------------------------------------
L2 Cache               1.9         500,000         0.048 ± 0.002        0.045 ± 0.002        0.011 ± 0.006
L2 Cache               3.8       1,000,000         0.103 ± 0.006        0.056 ± 0.002        0.024 ± 0.003
VRAM                  19.1       5,000,000         0.470 ± 0.014        0.101 ± 0.002        0.071 ± 0.002
VRAM                  38.1      10,000,000         0.914 ± 0.021        0.160 ± 0.004        0.130 ± 0.004
VRAM                 190.7      50,000,000         4.472 ± 0.045        0.629 ± 0.014        0.599 ± 0.016

RELATIVE SPEEDUP
Memory Location    Data Size   compile/PyTorch       CUDA/PyTorch       CUDA/compile
                        (MB)         (speedup)          (speedup)          (speedup)
--------------------------------------------------------------------------------------------------------------
L2 Cache               1.9        1.05 ± 0.05x        4.17 ± 2.21x        3.98 ± 2.10x
L2 Cache               3.8        1.84 ± 0.12x        4.37 ± 0.64x        2.38 ± 0.33x
VRAM                  19.1        4.65 ± 0.17x        6.62 ± 0.27x        1.43 ± 0.05x
VRAM                  38.1        5.73 ± 0.19x        7.05 ± 0.26x        1.23 ± 0.05x
VRAM                 190.7        7.11 ± 0.17x        7.47 ± 0.21x        1.05 ± 0.04x
```

**Key insights:**
- **Custom CUDA dominates at small sizes**: 4.2-4.4x faster than raw PyTorch when data fits in L2 cache
- **torch.compile catches up at large sizes**: At 50M elements, both hit memory bandwidth ceiling (1.05x difference)
- **Custom CUDA still wins mid-range**: At 5-10M elements, Custom CUDA is 1.2-1.4x faster than torch.compile

### add_mul_exp: `exp((x + y) * 2)`

```
RAW EXECUTION TIME
Memory Location    Data Size      # Elements            PyTorch      torch.compile        Custom CUDA
                        (MB)                               (ms)               (ms)               (ms)
--------------------------------------------------------------------------------------------------------------
L2 Cache               1.1         100,000         0.029 ± 0.003        0.047 ± 0.003        0.012 ± 0.001
L2 Cache               3.4         300,000         0.037 ± 0.002        0.063 ± 0.003        0.020 ± 0.001
VRAM                  11.4       1,000,000         0.080 ± 0.002        0.074 ± 0.002        0.043 ± 0.002
VRAM                  57.2       5,000,000         0.355 ± 0.011        0.191 ± 0.005        0.158 ± 0.004
VRAM                 228.9      20,000,000         1.365 ± 0.026        0.622 ± 0.015        0.589 ± 0.039

RELATIVE SPEEDUP
Memory Location    Data Size   compile/PyTorch       CUDA/PyTorch       CUDA/compile
                        (MB)         (speedup)          (speedup)          (speedup)
--------------------------------------------------------------------------------------------------------------
L2 Cache               1.1        0.62 ± 0.07x        2.37 ± 0.36x        3.83 ± 0.53x
L2 Cache               3.4        0.60 ± 0.05x        1.87 ± 0.18x        3.15 ± 0.27x
VRAM                  11.4        1.08 ± 0.05x        1.84 ± 0.10x        1.71 ± 0.09x
VRAM                  57.2        1.86 ± 0.08x        2.25 ± 0.09x        1.21 ± 0.05x
VRAM                 228.9        2.19 ± 0.07x        2.32 ± 0.16x        1.06 ± 0.07x
```

**Key insights:**
- **torch.compile overhead at small sizes**: torch.compile is slower than raw PyTorch at 100K-300K elements due to graph execution overhead
- **Custom CUDA dominates small sizes**: 2.4-3.8x faster than torch.compile when data fits in L2 cache (100K-300K elements)
- **Performance converges at large sizes**: At 20M elements, Custom CUDA is only 1.06x faster than torch.compile - both are memory-bandwidth-limited
- **Custom CUDA consistently wins**: 1.84-2.32x faster than raw PyTorch across all sizes

**Why is it faster?**

Modern GPUs are **memory-bound**. Memory access is ~400x slower than register operations. Kernel fusion:
- **Reduces kernel launches**: 3-5 kernels → 1 kernel
- **Eliminates intermediate memory**: Keeps values in registers instead of VRAM
- **Improves cache utilization**: Better temporal locality at small sizes

**Performance metrics:**
- **Numerical accuracy**: All implementations produce identical results (< 1e-5 error)
- **Memory traffic**: Custom CUDA reduces memory operations by 2.3-5x
- **Cache effects**: Performance advantage is highest when data fits in L2 cache, converges when memory-bandwidth-limited

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

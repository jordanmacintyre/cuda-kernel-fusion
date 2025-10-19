# CUDA Kernel Fusion

Educational examples demonstrating **CUDA kernel fusion** to optimize GPU performance by reducing memory bandwidth bottlenecks. Achieves **2-3x speedup** over PyTorch by keeping intermediate values in GPU registers instead of global memory.

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
float a = x[idx] + y[idx];      // register (fast!)
float b = a * 2.0f;              // register (fast!)
output[idx] = expf(b);           // write to memory
```

**Result:** 2.3x speedup, 99% efficiency, < 1e-6 relative error

## Why It Works

Modern GPUs are **memory-bound**:
- **Compute:** 20 TFLOPS
- **Memory:** 448 GB/s ← **bottleneck**

Memory access is **400x slower** than register access. Fusion minimizes memory traffic:
- **PyTorch:** 7 memory ops → 267 MB transferred
- **Fused:** 3 memory ops → 114 MB transferred
- **Result:** 2.3x faster ⚡

## Quick Start

```bash
# 1. Setup environment
conda create -n cuda-fusion python=3.12
conda activate cuda-fusion
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c nvidia cuda-toolkit ninja
pip install pytest tqdm numpy

# 2. Install package (editable mode for development)
pip install --no-build-isolation -e .

# 3. Run tests
pytest tests/ -v

# 4. Run benchmarks
python benchmarks/benchmark_add_mul_exp.py
```

### Optional: Set GPU Architecture for Faster Compilation

```bash
# Find your GPU's compute capability
python -c "import torch; print(torch.cuda.get_device_capability())"

# Set it in your environment (example for RTX 3070/3080 - compute 8.6)
export TORCH_CUDA_ARCH_LIST='8.6'
```

## Usage

```python
import torch
from cuda_ops import add_mul_exp

# Create input tensors on GPU
x = torch.randn(1_000_000, device='cuda')
y = torch.randn(1_000_000, device='cuda')

# Fused operation: exp((x + y) * 2)
result = add_mul_exp(x, y)  # 2.3x faster than PyTorch!

# Compare with PyTorch
pytorch_result = torch.exp((x + y) * 2)
assert torch.allclose(result, pytorch_result, rtol=1e-5)
```

## Project Structure

```
cuda-kernel-fusion/
├── cuda_ops/                      # CUDA operations package
│   ├── __init__.py               # Package initialization
│   ├── add_mul_exp.py            # Python wrapper with input validation
│   └── csrc/
│       └── add_mul_exp.cu        # CUDA kernel implementation
├── tests/                         # Comprehensive test suite
│   └── test_add_mul_exp.py       # 19 tests (correctness, accuracy, edge cases)
├── benchmarks/                    # Performance analysis
│   └── benchmark_add_mul_exp.py  # Detailed profiling and comparison
├── pyproject.toml                # Project configuration & pytest settings
└── README.md                     # This file
```

## Testing

The project includes a comprehensive test suite with **19 tests** covering correctness, numerical accuracy, and edge cases.

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Skip slow tests (runs 16/19 tests)
pytest tests/ -m "not slow"

# Run only numerical accuracy tests (3 tests)
pytest tests/ -m "numerical"

# Run only CUDA tests (all 19 tests)
pytest tests/ -m "cuda"

# Run specific test
pytest tests/test_add_mul_exp.py::TestAddMulExp::test_basic_correctness -v
```

### Test Coverage

- ✅ **Correctness:** Basic operations, known values, various sizes
- ✅ **Input validation:** CPU tensors, shape mismatches, dtype checks
- ✅ **Edge cases:** Zero inputs, large values, numerical stability
- ✅ **Accuracy:** Max/mean relative error < 1e-5
- ✅ **Performance:** 10M element stress tests

All tests pass with **< 1e-5 relative error** compared to PyTorch.

## Benchmarking

Run comprehensive performance analysis:

```bash
python benchmarks/benchmark_add_mul_exp.py
```

**Example output (RTX 3070, 10M elements):**

```
PyTorch (unfused):   0.686ms
CUDA (fused):        0.297ms
Speedup:             2.31x (99.2% of theoretical 2.33x)
Max relative error:  8.13e-07 ✓
```

The benchmark provides:
- ⏱️ **Detailed timing** with warmup and statistics
- 🔬 **Numerical accuracy** analysis (absolute/relative errors)
- 📊 **Memory bandwidth** calculations
- 🧪 **PyTorch profiler** kernel-level timing

## Performance Analysis

### Memory Hierarchy (Speed)

```
Registers:       1 cycle    ← Keep data here!
Shared memory:   20 cycles
L2 cache:        200 cycles
Global memory:   400 cycles ← Avoid round-trips
```

### Memory Traffic Comparison

| Version | Operations | Reads | Writes | Total Traffic |
|---------|-----------|-------|--------|---------------|
| PyTorch (unfused) | 3 kernels | x, y, a, b | a, b, c | 267 MB |
| CUDA (fused) | 1 kernel | x, y | c | 114 MB |
| **Reduction** | **2x fewer** | **2x fewer** | **3x fewer** | **2.3x less** |

### Key Concepts

#### Thread Indexing
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// Block 0, Thread 0:   idx = 0
// Block 0, Thread 255: idx = 255
// Block 1, Thread 0:   idx = 256
```

#### Kernel Launch
```cuda
int threads = 256;
int blocks = (size + threads - 1) / threads;  // Ceiling division
my_kernel<<<blocks, threads>>>(x, y, output, size);
```

## Adding New Fused Operations

### 1. Create CUDA Kernel

Create `cuda_ops/csrc/your_op.cu`:

```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void your_op_kernel(
    const float* x,
    const float* y,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Your fused operations here
        float a = x[idx] + y[idx];
        float b = a * 2.0f;
        output[idx] = expf(b);
    }
}

torch::Tensor your_op_cuda(torch::Tensor x, torch::Tensor y) {
    auto output = torch::empty_like(x);

    int threads = 256;
    int blocks = (x.numel() + threads - 1) / threads;

    your_op_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel()
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("your_op_cuda", &your_op_cuda, "Your operation (CUDA)");
}
```

### 2. Create Python Wrapper

Create `cuda_ops/your_op.py`:

```python
import torch

# Import compiled CUDA extension
from cuda_ops._C import your_op_cuda

def your_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Your operation with input validation."""

    # Input validation
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")

    if x.dtype != torch.float32:
        raise ValueError("Only float32 supported")

    return your_op_cuda(x, y)
```

### 3. Update Package

Add to `cuda_ops/__init__.py`:

```python
from .your_op import your_op
__all__ = ["add_mul_exp", "your_op"]
```

Update `pyproject.toml` to compile new kernel.

### 4. Add Tests

Create `tests/test_your_op.py` following the pattern in `test_add_mul_exp.py`.

## Good Practice Operations

Operations that benefit from fusion:

1. **Activation functions:**
   - `sigmoid(x * w + b)` - Fuse multiply, add, sigmoid
   - `gelu(x)` - Custom activation
   - `swish(x) = x * sigmoid(x)` - Fuse multiply and sigmoid

2. **Normalization:**
   - `layer_norm(x)` - Fuse mean, variance, normalize
   - `rms_norm(x)` - Root mean square normalization

3. **Element-wise chains:**
   - `(x + y) * z + w` - Multiple arithmetic ops
   - `dropout(relu(x))` - Combine activation and dropout

## Troubleshooting

| Error | Solution |
|-------|----------|
| `nvcc not found` | `conda install -c nvidia cuda-toolkit` |
| `ninja not found` | `conda install ninja` |
| Slow compilation | Set `TORCH_CUDA_ARCH_LIST='8.6'` for your GPU |
| Import error | Run `pip install --no-build-isolation -e .` |
| Tests fail | Ensure CUDA is available: `torch.cuda.is_available()` |
| Memory error | Reduce test sizes or use smaller batches |

### VS Code IntelliSense Setup

To remove red squiggles in CUDA files, create `.vscode/c_cpp_properties.json`:

```json
{
    "configurations": [{
        "name": "Linux",
        "includePath": [
            "${workspaceFolder}/**",
            "${env:HOME}/miniconda3/envs/cuda-fusion/targets/x86_64-linux/include",
            "${env:HOME}/miniconda3/envs/cuda-fusion/lib/python3.12/site-packages/torch/include",
            "${env:HOME}/miniconda3/envs/cuda-fusion/lib/python3.12/site-packages/torch/include/torch/csrc/api/include"
        ],
        "defines": ["__CUDACC__", "__NVCC__"],
        "compilerPath": "${env:HOME}/miniconda3/envs/cuda-fusion/bin/nvcc",
        "cStandard": "c17",
        "cppStandard": "c++17"
    }],
    "version": 4
}
```

Adjust paths based on your conda environment location.

## Performance Tips

1. **Use appropriate block sizes:** 256 threads/block is a good default
2. **Coalesce memory access:** Access consecutive memory locations in warps
3. **Minimize global memory access:** Keep intermediate values in registers
4. **Use `__restrict__` pointers:** Helps compiler optimize
5. **Profile with `nsys`:** NVIDIA Nsight Systems for detailed profiling
6. **Use `--use_fast_math`:** Trade precision for speed when appropriate

## Resources

- 📖 [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- 🔧 [PyTorch Custom C++/CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- 💡 [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- 📊 [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) - Kernel profiler

## Contributing

Contributions welcome! Please:

1. Add tests for new operations
2. Run `pytest tests/` to ensure all tests pass
3. Run benchmarks to verify performance gains
4. Update documentation

## License

MIT - See LICENSE file for details

---

**Author:** Jordan MacIntyre
**Purpose:** Educational demonstration of CUDA kernel fusion techniques
**GPU Tested:** RTX 3070 (compute capability 8.6)

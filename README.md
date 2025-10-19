# CUDA Kernel Fusion

Educational examples demonstrating **CUDA kernel fusion** to optimize GPU performance by reducing memory bandwidth bottlenecks. Achieves **2-3x speedup** over PyTorch by keeping intermediate values in GPU registers.

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

## Usage

```python
import torch
from cuda_ops import add_mul_exp

x = torch.randn(1_000_000, device='cuda')
y = torch.randn(1_000_000, device='cuda')

# Fused operation: exp((x + y) * 2)
result = add_mul_exp(x, y)  # 2.3x faster than PyTorch!
```

## Performance

**RTX 3070, 10M elements:**
```
PyTorch (unfused):   0.686ms
CUDA (fused):        0.297ms
Speedup:             2.31x (99.2% efficiency)
Max relative error:  8.13e-07 ✓
```

**Why is it faster?**

Modern GPUs are **memory-bound**. Memory access is 400x slower than register operations.

| Version | Memory Operations | Total Traffic | Time |
|---------|------------------|---------------|------|
| PyTorch | 7 (read x,y,a,b; write a,b,c) | 267 MB | 0.686ms |
| CUDA Fused | 3 (read x,y; write c) | 114 MB | 0.297ms |
| **Savings** | **4 fewer ops (57%)** | **153 MB (57%)** | **2.3x faster** |

Intermediate values `a` and `b` stay in registers → **no memory round-trips**.

## Project Structure

```
cuda-kernel-fusion/
├── cuda_ops/                      # CUDA operations package
│   ├── add_mul_exp.py            # Python wrapper with validation
│   └── csrc/add_mul_exp.cu       # CUDA kernel implementation
├── tests/                         # 19 tests (correctness, accuracy, edge cases)
│   └── test_add_mul_exp.py
├── benchmarks/                    # Performance analysis
│   ├── utils/                    # Reusable benchmarking framework
│   ├── bench_add_mul_exp.py      # Example benchmark
│   └── bench_template.py         # Template for new kernels
├── pyproject.toml                # Config & dependencies
└── README.md
```

## Testing

```bash
# Run all 19 tests
pytest tests/ -v

# Skip slow tests (16 tests)
pytest tests/ -m "not slow"

# Run only numerical accuracy tests
pytest tests/ -m "numerical"
```

**Test coverage:**
- ✅ Correctness (basic ops, known values, various sizes)
- ✅ Input validation (CPU tensors, shape/dtype checks)
- ✅ Edge cases (zeros, large values, numerical stability)
- ✅ Accuracy (< 1e-5 relative error)

## Benchmarking

```bash
# Run comprehensive benchmark
python benchmarks/bench_add_mul_exp.py
```

**Output includes:**
- ⏱️ Detailed timing with warmup and statistics
- 🔬 Numerical accuracy analysis
- 📊 Memory bandwidth calculations
- 🧪 PyTorch profiler kernel-level details

See `benchmarks/README.md` for details on the benchmarking framework.

## Adding New Operations

### 1. Create CUDA Kernel

`cuda_ops/csrc/your_op.cu`:
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

### 2. Create Python Wrapper

`cuda_ops/your_op.py`:
```python
import torch
from cuda_ops._C import your_op_cuda

def your_op(x: torch.Tensor, y: torch.Tensor) -> torch::Tensor:
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("Inputs must be CUDA tensors")
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    return your_op_cuda(x, y)
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

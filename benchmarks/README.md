# Benchmarks

Comprehensive performance and accuracy analysis framework for CUDA kernel fusion operations. Provides timing statistics, numerical accuracy metrics, memory traffic analysis with **roofline model efficiency**, and profiling for comparing fused CUDA kernels against PyTorch baselines.

## Contents

This directory contains:
- **Benchmark scripts**: Complete benchmarks for each implemented kernel
  - [bench_add_mul_exp.py](bench_add_mul_exp.py) - Element-wise fusion benchmark
  - [bench_quantize_int8.py](bench_quantize_int8.py) - INT8 quantization benchmark
- **Template**: [bench_template.py](bench_template.py) - Starting point for new benchmarks
- **Utilities**: Reusable benchmarking framework in [utils/](utils/)
  - [performance.py](utils/performance.py) - Timing, profiling, memory analysis
  - [analysis.py](utils/analysis.py) - Numerical accuracy comparison

## Quick Start

```bash
# Run element-wise fusion benchmark
python benchmarks/bench_add_mul_exp.py

# Run INT8 quantization benchmark
python benchmarks/bench_quantize_int8.py
```

**Sample output:**
```
======================================================================
TIMING RESULTS
======================================================================

PyTorch:
  Mean:   0.690 ± 0.020 ms
  Median: 0.687 ms
  Range:  [0.681, 0.788] ms

CUDA:
  Mean:   0.297 ± 0.010 ms
  Median: 0.295 ms
  Range:  [0.291, 0.361] ms

  Speedup: 2.32x

======================================================================
MEMORY TRAFFIC ANALYSIS
======================================================================
Speedup Analysis:
  Memory reduction:       2.33x
  Kernel reduction:       3/1 = 3.00x
  Actual speedup:         2.32x

Roofline Analysis:
  Baseline arithmetic intensity:  0.107 FLOPs/byte
  Baseline bottleneck:            memory
  Baseline efficiency:            98.0%
  Optimized arithmetic intensity: 0.250 FLOPs/byte
  Optimized bottleneck:           memory
  Optimized efficiency:           97.5%

======================================================================
NUMERICAL ACCURACY ANALYSIS
======================================================================
Relative Differences:
  Max:    8.861581e-07

Passes torch.allclose():
  rtol=1e-5: ✓
```

## Creating a New Benchmark

### 1. Copy the Template

```bash
cp benchmarks/bench_template.py benchmarks/bench_your_op.py
```

### 2. Implement Your Functions

```python
def pytorch_baseline(x, y):
    """Your operation using PyTorch."""
    return torch.sigmoid(x * y + 1)  # Example

def cuda_optimized(x, y):
    """Your CUDA kernel."""
    from ops.cuda import your_op
    return your_op(x, y)
```

### 3. Update Memory Operation Counts

Count the number of tensor reads/writes for accurate bandwidth analysis. For each PyTorch operation, count how many times tensors are read from or written to global memory:

```python
# Basic memory analysis (without roofline)
memory_stats = analyze_memory_traffic(
    tensor_size=size,
    baseline_reads=3,   # PyTorch: count all tensor reads across all ops
    baseline_writes=2,  # PyTorch: count all tensor writes across all ops
    optimized_reads=2,  # CUDA: read x, y only
    optimized_writes=1, # CUDA: write output only
    baseline_time_ms=baseline_result.mean_time_ms,
    optimized_time_ms=optimized_result.mean_time_ms,
)

# Advanced analysis with roofline model (recommended)
memory_stats = analyze_memory_traffic(
    tensor_size=size,
    baseline_reads=3,
    baseline_writes=2,
    optimized_reads=2,
    optimized_writes=1,
    baseline_time_ms=baseline_result.mean_time_ms,
    optimized_time_ms=optimized_result.mean_time_ms,
    # Roofline model parameters (optional but recommended)
    baseline_flops=size * 3,      # Total FLOPs for baseline
    optimized_flops=size * 3,     # Total FLOPs for optimized
    gpu_specs={
        'peak_bandwidth': 414e9,   # Measured GPU bandwidth (GB/s)
        'peak_flops': 20.3e12,     # GPU peak FLOPs (FLOPS)
    }
)
```

**Example:** For `clamp(round(x / scale + zero_point), -128, 127)`:
- PyTorch: 5 ops → 5 reads + 5 writes = 10 memory operations
- CUDA: 1 op → 1 read + 1 write = 2 memory operations

### 4. Run Your Benchmark

```bash
python benchmarks/bench_your_op.py
```

## Using the Benchmarking Utilities

### Simple Comparison

```python
from utils import compare_implementations

baseline_result, optimized_result = compare_implementations(
    baseline_func=pytorch_impl,
    optimized_func=cuda_impl,
    args=(x, y),
    baseline_name="PyTorch",
    optimized_name="CUDA",
    iterations=100
)

print(f"Speedup: {baseline_result.mean_time_ms / optimized_result.mean_time_ms:.2f}x")
```

### Numerical Accuracy

```python
from utils import analyze_numerical_accuracy

metrics = analyze_numerical_accuracy(
    baseline=pytorch_result,
    optimized=cuda_result,
    verbose=True
)

print(f"Max relative error: {metrics.max_rel_diff:.2e}")
print(f"Passes 1e-5: {metrics.passes_1e5}")
```

### Quick Accuracy Check

```python
from utils import compare_accuracy_quick

passes, abs_err, rel_err = compare_accuracy_quick(
    baseline=pytorch_result,
    optimized=cuda_result,
    rtol=1e-5
)

if not passes:
    print(f"Failed! Max abs error: {abs_err:.2e}, rel error: {rel_err:.2e}")
```

### Memory Traffic Analysis

```python
from utils import analyze_memory_traffic, print_memory_analysis

stats = analyze_memory_traffic(
    tensor_size=10_000_000,
    baseline_reads=4,
    baseline_writes=3,
    optimized_reads=2,
    optimized_writes=1,
    baseline_time_ms=0.686,
    optimized_time_ms=0.297
)

print_memory_analysis(stats)
```

Output (with roofline model):
```
Memory Traffic Analysis:
  Baseline:
    Memory operations: 7 (4 reads + 3 writes)
    Total traffic:     267.03 MB
    Bandwidth:         378.81 GB/s

  Optimized:
    Memory operations: 3 (2 reads + 1 writes)
    Total traffic:     114.44 MB
    Bandwidth:         399.72 GB/s

  Speedup Analysis:
    Memory reduction:       2.33x
    Kernel reduction:       3/1 = 3.00x
    Actual speedup:         2.32x

  Roofline Analysis:
    Baseline arithmetic intensity:  0.107 FLOPs/byte
    Baseline bottleneck:            memory
    Baseline efficiency:            98.0%
    Optimized arithmetic intensity: 0.250 FLOPs/byte
    Optimized bottleneck:           memory
    Optimized efficiency:           97.5%
```

### GPU Specifications for Roofline

Measure your GPU's actual bandwidth and peak FLOPs:

```python
from utils import measure_gpu_specs

gpu_specs = measure_gpu_specs(
    size_mb=1000,      # Test with 1GB of data
    num_iterations=100,
    verbose=True
)

print(f"Peak bandwidth: {gpu_specs['peak_bandwidth']/1e9:.1f} GB/s")
print(f"Peak FLOPs: {gpu_specs['peak_flops']/1e12:.1f} TFLOPS")
```

**Note**: This measures **copy bandwidth**, not compute bandwidth. For production analysis, consider using compute-based measurements (FMA operations).

### PyTorch Profiler

```python
from utils import profile_with_pytorch_profiler

profile_with_pytorch_profiler([
    (pytorch_impl, "PyTorch", (x, y)),
    (cuda_impl, "CUDA", (x, y))
], iterations=10)
```

## Utility Functions Reference

### `utils/performance.py`

| Function | Description | Returns |
|----------|-------------|---------|
| `benchmark_function()` | Benchmark single function with warmup | `BenchmarkResult` with timing stats |
| `compare_implementations()` | Compare baseline vs optimized | Tuple of two `BenchmarkResult`s |
| `measure_gpu_specs()` | Measure actual GPU bandwidth and estimate peak FLOPs | Dict with `peak_bandwidth`, `peak_flops` |
| `roofline_efficiency()` | Calculate roofline model efficiency | `(efficiency, bottleneck, cache_benefit)` tuple |
| `analyze_memory_traffic()` | Calculate bandwidth, efficiency, roofline metrics | Dict with memory analysis |
| `print_memory_analysis()` | Pretty-print memory stats with roofline | None (prints to stdout) |
| `profile_with_pytorch_profiler()` | Profile using PyTorch profiler | None (prints profiler output) |

### `utils/analysis.py`

| Function | Description | Returns |
|----------|-------------|---------|
| `analyze_numerical_accuracy()` | Comprehensive accuracy analysis | `AccuracyMetrics` object |
| `compare_accuracy_quick()` | Quick pass/fail accuracy check | `(passes, max_abs_err, max_rel_err)` |
| `assert_accuracy()` | Assert accuracy or raise error | None (raises on failure) |

See the source files for detailed parameter documentation.

## Understanding Performance Metrics

### Roofline Model

The roofline model determines if your kernel is **compute-bound** or **memory-bound** by comparing arithmetic intensity against the ridge point:

- **Arithmetic Intensity**: FLOPs per byte (FLOPs/byte)
- **Ridge Point**: `peak_flops / peak_bandwidth` (e.g., ~49 FLOPs/byte for RTX 3070)
- **Bottleneck**:
  - If AI < Ridge Point → **memory-bound** (limited by bandwidth)
  - If AI ≥ Ridge Point → **compute-bound** (limited by FLOPs)

**Roofline Efficiency** measures how close your kernel is to theoretical peak performance:
- **100%**: Perfect efficiency, achieving theoretical maximum
- **90-99%**: Excellent performance
- **< 90%**: Room for optimization

### Cache Benefit

When a kernel runs **faster than theoretical predictions** based on DRAM bandwidth alone, it indicates **L2 cache benefits**:

- **Cache benefit = 1.0x**: Performance matches DRAM-only theory
- **Cache benefit > 1.0x**: L2 cache is helping! (GOOD)
- **Example**: 1.51x cache benefit means the kernel is 51% faster than theory predicts

This happens when:
- Data fits in L2 cache (6MB on RTX 3070)
- Good temporal locality (reusing data)
- Effective spatial locality (coalesced access)

**Why report separately?** Efficiency is capped at 100% (can't exceed theoretical maximum), but cache benefit shows how much faster you are than the simple DRAM-only model.

## Best Practices

### 1. Always Include Both Analyses

```python
# Performance
baseline_result, optimized_result = compare_implementations(...)

# Accuracy
accuracy_metrics = analyze_numerical_accuracy(
    baseline_result.result,
    optimized_result.result
)
```

### 2. Set Appropriate Tolerances

- `rtol=1e-5` is typical for float32
- `rtol=1e-7` for float64
- Adjust based on your operation's numerical sensitivity

### 3. Understanding Memory Operations

Count only **global memory** accesses (reads from/writes to DRAM):
- **Reads**: Tensor loaded from global memory
- **Writes**: Tensor stored to global memory
- **Don't count**: Register operations, L1/L2 cache hits, shared memory

**Example:** `exp((x + y) * 2)`

PyTorch (3 separate kernels):
1. `a = x + y`: read x, y → write a (2 reads, 1 write)
2. `b = a * 2`: read a → write b (1 read, 1 write)
3. `c = exp(b)`: read b → write c (1 read, 1 write)
- **Total: 4 reads + 3 writes = 7 operations**

CUDA (1 fused kernel):
- Read x, y → compute in registers → write c
- **Total: 2 reads + 1 write = 3 operations**

### 4. Use Sufficient Warmup

- Default `warmup=10` is usually sufficient
- Increase for:
  - First run after compilation
  - Unstable timing results
  - Very fast operations (< 0.1ms)

### 5. Profile for Details

Use PyTorch profiler to understand kernel-level behavior:
- Kernel launch overhead
- Memory transfer time
- Actual kernel execution time

## Troubleshooting

**Unstable timings:**
- Increase warmup iterations
- Increase benchmark iterations
- Check GPU isn't throttling (thermals)
- Close other GPU applications

**Accuracy failures:**
- Check operation is mathematically correct
- Verify input ranges don't cause overflow
- Use double precision for reference
- Check for NaN/Inf propagation

**Low speedup:**
- Verify memory operation counts
- Check kernel isn't limited by other factors (divergence, atomics)
- Profile to identify bottlenecks
- Ensure data is actually on GPU

## Example Benchmarks

- [bench_add_mul_exp.py](bench_add_mul_exp.py) - Element-wise fusion: `exp((x + y) * 2)`
  - Demonstrates reducing 3 kernels to 1
  - Shows 2.32x speedup with 98% roofline efficiency
  - High numerical accuracy (< 1e-6 relative error)

- [bench_quantize_int8.py](bench_quantize_int8.py) - INT8 quantization
  - Demonstrates reducing 5 kernels to 1
  - Shows 7.14x speedup with 100% roofline efficiency + 1.51x cache benefit
  - Exact integer matching (zero error)

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

Each benchmark tests **three implementations** (PyTorch, torch.compile, Custom CUDA) across **five data sizes** to analyze cache effects and memory bandwidth limits.

**Sample output:**
```
==============================================================================================================================
PERFORMANCE SUMMARY
==============================================================================================================================

RAW EXECUTION TIME
Memory Location       Data Size      # Elements            PyTorch      torch.compile        Custom CUDA
                           (MB)                               (ms)               (ms)               (ms)
------------------------------------------------------------------------------------------------------------------------------
L2 Cache                    1.9         500,000      0.048 ± 0.002      0.045 ± 0.002      0.011 ± 0.006
L2 Cache                    3.8       1,000,000      0.103 ± 0.006      0.056 ± 0.002      0.024 ± 0.003
VRAM                       19.1       5,000,000      0.470 ± 0.014      0.101 ± 0.002      0.071 ± 0.002
VRAM                       38.1      10,000,000      0.914 ± 0.021      0.160 ± 0.004      0.130 ± 0.004
VRAM                      190.7      50,000,000      4.472 ± 0.045      0.629 ± 0.014      0.599 ± 0.016

RELATIVE SPEEDUP
Memory Location       Data Size        compile/PyTorch           CUDA/PyTorch           CUDA/compile
                           (MB)              (speedup)              (speedup)              (speedup)
------------------------------------------------------------------------------------------------------------------------------
L2 Cache                    1.9          1.05 ± 0.05 x          4.17 ± 2.21 x          3.98 ± 2.10 x
L2 Cache                    3.8          1.84 ± 0.12 x          4.37 ± 0.64 x          2.38 ± 0.33 x
VRAM                       19.1          4.65 ± 0.17 x          6.62 ± 0.27 x          1.43 ± 0.05 x
VRAM                       38.1          5.73 ± 0.19 x          7.05 ± 0.26 x          1.23 ± 0.05 x
VRAM                      190.7          7.11 ± 0.17 x          7.47 ± 0.21 x          1.05 ± 0.04 x

==============================================================================================================================
Numerical Accuracy: PASS - All implementations match
==============================================================================================================================
```

**What the benchmark shows:**
- **L2 Cache data (2-4 MB)**: Custom CUDA is 4.2-4.4x faster than raw PyTorch - shows register and cache efficiency
- **Small VRAM data (19-38 MB)**: Both optimizations excel, 6-7x speedup over raw PyTorch
- **Large VRAM data (190 MB)**: Memory bandwidth limited - both hit the same 7.5x ceiling (1.05x difference)

**Test scenarios:**
- **L2 Cache (500K-1M elements)**: Fits in GPU L2 cache (~4 MB), tests register efficiency and cache utilization
- **VRAM (5M-50M elements)**: Exceeds L2 cache, tests memory bandwidth optimization and kernel fusion benefits

**Understanding the output:**
- **Raw execution time**: Mean ± standard deviation (ms) from 100 iterations
- **Relative speedup**: Calculated speedup with error propagation: σ = (A/B) × √[(σ_A/A)² + (σ_B/B)²]
- **Memory Location**: Whether data fits in L2 Cache or requires VRAM access
- **Data Size (MB)**: Calculated memory per tensor (float32 = 4 bytes/element)

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

### 3. Run Your Benchmark

```bash
python benchmarks/bench_your_op.py
```

## Using the Benchmarking Utilities

### Three-Way Comparison

The benchmarks compare PyTorch, torch.compile, and custom CUDA across multiple data sizes:

```python
from utils import compare_three_implementations

pytorch_result, compiled_result, cuda_result = compare_three_implementations(
    baseline_func=pytorch_impl,
    compiled_func=torch.compile(pytorch_impl),
    optimized_func=cuda_impl,
    args=(x, y),
    baseline_name="PyTorch",
    compiled_name="torch.compile",
    optimized_name="Custom CUDA",
    warmup=10,
    iterations=100
)

# Extract timing for summary
pytorch_time = pytorch_result.mean_time_ms
compiled_time = compiled_result.mean_time_ms
cuda_time = cuda_result.mean_time_ms

print(f"Speedup (CUDA vs PyTorch): {pytorch_time / cuda_time:.2f}x")
print(f"Speedup (CUDA vs compile): {compiled_time / cuda_time:.2f}x")
```

### Numerical Accuracy

The benchmarks automatically check numerical accuracy between all implementations:

```python
from utils import analyze_numerical_accuracy

# Compare PyTorch vs Custom CUDA
metrics = analyze_numerical_accuracy(
    baseline=pytorch_result.result,
    optimized=cuda_result.result,
    verbose=False  # Set True for detailed output
)

print(f"Max absolute error: {metrics.max_abs_diff:.2e}")
print(f"Max relative error: {metrics.max_rel_diff:.2e}")
print(f"Passes 1e-5 tolerance: {metrics.passes_1e5}")
```

**Memory-efficient design:**
- Only computes max and mean errors (not median/std) to save memory
- Skips NaN/Inf checks for large tensors (200M elements)
- Uses scalar comparisons instead of `torch.allclose()` to avoid OOM
- Suitable for benchmarking 200M+ element tensors on 8GB GPUs

### Advanced Analysis (Optional)

The benchmarking utilities also support detailed memory traffic and roofline analysis, though these are not included in the default benchmark output. These are useful for deep performance investigation:

**Available advanced utilities:**
- `analyze_memory_traffic()` - Calculate bandwidth, memory reduction, and roofline efficiency
- `measure_gpu_specs()` - Measure actual GPU bandwidth and peak FLOPs
- `roofline_efficiency()` - Determine if kernel is compute-bound or memory-bound

See the utility functions reference below for details on these advanced features.

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
| `compare_three_implementations()` | Compare PyTorch, torch.compile, and custom CUDA | Tuple of three `BenchmarkResult`s |
| `measure_gpu_specs()` | Measure actual GPU bandwidth and estimate peak FLOPs | Dict with `peak_bandwidth`, `peak_flops` |
| `roofline_efficiency()` | Calculate roofline model efficiency | `(efficiency, bottleneck, cache_benefit)` tuple |
| `analyze_memory_traffic()` | Calculate bandwidth, efficiency, roofline metrics | Dict with memory analysis |
| `print_memory_analysis()` | Pretty-print memory stats with roofline | None (prints to stdout) |
| `profile_with_pytorch_profiler()` | Profile using PyTorch profiler | None (prints profiler output) |

### `utils/analysis.py`

| Function | Description | Returns |
|----------|-------------|---------|
| `analyze_numerical_accuracy()` | Memory-efficient accuracy analysis | `AccuracyMetrics` object |

**Note:** The analysis utilities have been optimized for memory efficiency to handle 200M+ element tensors on 8GB GPUs. This includes:
- Simplified metrics (max/mean only, no median/std)
- Skipped special value checks (NaN/Inf)
- Scalar-based tolerance checks (no intermediate tensors)

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

When a kernel runs **faster than theoretical predictions** based on VRAM bandwidth alone, it indicates **L2 cache benefits**:

- **Cache benefit = 1.0x**: Performance matches VRAM-only theory
- **Cache benefit > 1.0x**: L2 cache is helping! (GOOD)
- **Example**: 1.51x cache benefit means the kernel is 51% faster than theory predicts

This happens when:
- Data fits in L2 cache (~4MB on RTX 3070)
- Good temporal locality (reusing data)
- Effective spatial locality (coalesced access)

**Why report separately?** Efficiency is capped at 100% (can't exceed theoretical maximum), but cache benefit shows how much faster you are than the simple VRAM-only model.

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

### 3. Use Sufficient Warmup

- Default `warmup=10` is usually sufficient
- Increase for:
  - First run after compilation
  - Unstable timing results
  - Very fast operations (< 0.1ms)

### 4. Profile for Details

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
- Profile to identify bottlenecks
- Check kernel isn't limited by other factors (divergence, atomics)
- Ensure data is actually on GPU
- Compare across different data sizes (L2 Cache vs VRAM)

## Example Benchmarks

- [bench_add_mul_exp.py](bench_add_mul_exp.py) - Element-wise fusion: `exp((x + y) * 2)`
  - Demonstrates kernel fusion reducing 3 PyTorch operations to 1 CUDA kernel
  - **L2 Cache**: 1.84x speedup over raw PyTorch (1M elements), 1.71x vs torch.compile
  - **VRAM**: 2.32x speedup over raw PyTorch (20M elements), 1.06x vs torch.compile
  - High numerical accuracy (< 1e-6 relative error)

- [bench_quantize_int8.py](bench_quantize_int8.py) - INT8 quantization
  - Demonstrates kernel fusion reducing 5 PyTorch operations to 1 CUDA kernel
  - **L2 Cache**: 4.4x speedup over raw PyTorch (1M elements), 2.4x vs torch.compile
  - **VRAM**: 7.5x speedup over raw PyTorch (50M elements), 1.05x vs torch.compile
  - Exact integer matching (zero error)

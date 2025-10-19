# Benchmarks

Comprehensive performance and accuracy benchmarks for CUDA kernel fusion operations.

## Structure

```
benchmarks/
├── utils/                      # Reusable benchmarking utilities
│   ├── __init__.py
│   ├── performance.py         # Performance benchmarking functions
│   └── analysis.py            # Numerical accuracy analysis
├── bench_add_mul_exp.py       # Benchmark for add_mul_exp kernel
├── bench_template.py          # Template for new benchmarks
└── README.md                  # This file
```

## Quick Start

### Run a Benchmark

```bash
# Run the add_mul_exp benchmark
python benchmarks/bench_add_mul_exp.py
```

### Expected Output

```
PyTorch (unfused):   0.686 ± 0.012 ms
CUDA (fused):        0.297 ± 0.008 ms
Speedup:             2.31x
Max relative error:  8.13e-07 ✓
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
    from cuda_ops import your_op
    return your_op(x, y)
```

### 3. Update Memory Counts

Count the number of tensor reads/writes for accurate bandwidth analysis:

```python
memory_stats = analyze_memory_traffic(
    tensor_size=size,
    baseline_reads=3,   # PyTorch: read x, y, intermediate
    baseline_writes=2,  # PyTorch: write intermediate, output
    optimized_reads=2,  # CUDA: read x, y
    optimized_writes=1, # CUDA: write output
    baseline_time_ms=baseline_result.mean_time_ms,
    optimized_time_ms=optimized_result.mean_time_ms,
)
```

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

Output:
```
Memory Traffic Analysis:
  Baseline:
    Memory operations: 7 (4 reads + 3 writes)
    Total traffic:     267.03 MB
    Bandwidth:         381.25 GB/s

  Optimized:
    Memory operations: 3 (2 reads + 1 writes)
    Total traffic:     114.44 MB
    Bandwidth:         377.36 GB/s

  Efficiency:
    Theoretical speedup: 2.33x
    Actual speedup:      2.31x
    Efficiency:          99.1%
```

### PyTorch Profiler

```python
from utils import profile_with_pytorch_profiler

profile_with_pytorch_profiler([
    (pytorch_impl, "PyTorch", (x, y)),
    (cuda_impl, "CUDA", (x, y))
], iterations=10)
```

## API Reference

### performance.py

**`benchmark_function(func, args, name, warmup=10, iterations=100, verbose=True)`**
- Benchmark a single function with warmup
- Returns: `BenchmarkResult` with timing statistics

**`compare_implementations(baseline_func, optimized_func, args, ...)`**
- Compare two implementations
- Returns: `(baseline_result, optimized_result)`

**`analyze_memory_traffic(tensor_size, baseline_reads, baseline_writes, ...)`**
- Calculate memory traffic and bandwidth
- Returns: Dictionary with detailed analysis

**`print_memory_analysis(analysis)`**
- Pretty print memory analysis
- Input: Output from `analyze_memory_traffic()`

**`profile_with_pytorch_profiler(funcs_and_names, iterations=10)`**
- Profile using PyTorch's profiler
- Input: List of `(function, name, args)` tuples

### analysis.py

**`analyze_numerical_accuracy(baseline, optimized, verbose=True)`**
- Comprehensive accuracy analysis
- Returns: `AccuracyMetrics` with detailed comparison

**`compare_accuracy_quick(baseline, optimized, rtol=1e-5, atol=1e-8)`**
- Quick pass/fail check
- Returns: `(passes, max_abs_error, max_rel_error)`

**`assert_accuracy(baseline, optimized, rtol=1e-5, operation_name="Operation")`**
- Assert accuracy or raise detailed error
- Useful in test suites

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

### 3. Count Memory Operations Carefully

Memory operations include:
- **Reads**: Every time a tensor is loaded from global memory
- **Writes**: Every time a tensor is stored to global memory
- **Don't count**: Register operations, shared memory (for simple kernels)

Example for `exp((x + y) * 2)`:

**PyTorch (3 kernels):**
- Kernel 1: read x, y → write a (2 reads, 1 write)
- Kernel 2: read a → write b (1 read, 1 write)
- Kernel 3: read b → write c (1 read, 1 write)
- **Total: 4 reads, 3 writes**

**CUDA (1 kernel):**
- read x, y → write c (2 reads, 1 write)
- **Total: 2 reads, 1 write**

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

## Examples

See `bench_add_mul_exp.py` for a complete working example.

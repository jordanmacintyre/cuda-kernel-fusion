"""
Comprehensive CUDA Kernel Performance Analysis

This script compares:
1. Unfused PyTorch operations (3 separate GPU kernel launches)
2. Fused CUDA kernel (1 GPU kernel launch)

With detailed performance profiling and numerical accuracy analysis.
"""

import os

# Set CUDA architecture BEFORE importing torch
# Find your GPU's compute capability with:
#   python -c "import torch; print(torch.cuda.get_device_capability())"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"  # Adjust for your GPU

import sys
import time

import numpy as np
import torch
from tqdm import tqdm

print("=" * 70)
print("CUDA KERNEL FUSION: COMPREHENSIVE ANALYSIS")
print("=" * 70)

print("\n[SETUP] Loading CUDA extension...")
print("(This may take 10-60 seconds on first run - compiling CUDA code)")
sys.stdout.flush()

start_import = time.time()
from cuda_ops import add_mul_exp

elapsed_import = time.time() - start_import
print(f"✓ CUDA extension loaded in {elapsed_import:.1f}s")


def pytorch_unfused(x, y):
    """
    Unfused version using PyTorch operations.
    Each line triggers a separate CUDA kernel launch.
    """
    a = x + y  # Kernel 1: addition
    b = a * 2  # Kernel 2: multiplication
    c = torch.exp(b)  # Kernel 3: exponential
    return c


def cuda_fused(x, y):
    """
    Fused version using custom CUDA kernel.
    All operations in a single kernel launch.
    """
    return add_mul_exp(x, y)


def benchmark(func, x, y, name, warmup=10, iterations=100):
    """
    Benchmark a function with warmup and multiple iterations.
    Returns: (result, avg_time_ms, std_time_ms)
    """
    # Warmup
    for _ in tqdm(range(warmup), desc=f"  Warmup", leave=False, ncols=70):
        _ = func(x, y)
    torch.cuda.synchronize()

    # Collect multiple timings for statistics
    times = []
    result = None
    for _ in tqdm(range(iterations), desc=f"  Timing", leave=False, ncols=70):
        start = time.perf_counter()
        result = func(x, y)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # Convert to milliseconds
    print(f"  ✓ {name}: {times.mean():.3f} ± {times.std():.3f} ms")
    return result, times.mean(), times.std()


def analyze_numerical_accuracy(pytorch_result, cuda_result):
    """
    Comprehensive numerical accuracy analysis.
    """
    print("\n" + "=" * 70)
    print("NUMERICAL ACCURACY ANALYSIS")
    print("=" * 70)

    print("\nCalculating differences...")
    sys.stdout.flush()

    # Absolute differences
    abs_diff = torch.abs(pytorch_result - cuda_result)

    print("\nAbsolute Differences:")
    print(f"  Max:    {abs_diff.max():.6e}")
    print(f"  Mean:   {abs_diff.mean():.6e}")
    print(f"  Median: {abs_diff.median():.6e}")
    print(f"  Std:    {abs_diff.std():.6e}")

    # Relative differences (avoid division by zero)
    # Use max(|pytorch|, |cuda|) as denominator
    denominator = torch.maximum(torch.abs(pytorch_result), torch.abs(cuda_result))
    denominator = torch.clamp(denominator, min=1e-10)  # Avoid div by zero
    rel_diff = abs_diff / denominator

    print("\nRelative Differences:")
    print(f"  Max:    {rel_diff.max():.6e}")
    print(f"  Mean:   {rel_diff.mean():.6e}")
    print(f"  Median: {rel_diff.median():.6e}")
    print(f"  Std:    {rel_diff.std():.6e}")

    # Analyze the maximum difference location
    max_diff_idx = torch.argmax(abs_diff)

    print("\nMax Difference Location Analysis:")
    print(f"  Index:           {max_diff_idx.item()}")
    print(f"  PyTorch value:   {pytorch_result[max_diff_idx]:.10f}")
    print(f"  CUDA value:      {cuda_result[max_diff_idx]:.10f}")
    print(f"  Absolute diff:   {abs_diff[max_diff_idx]:.10f}")
    print(f"  Relative diff:   {rel_diff[max_diff_idx]:.6e}")

    # Distribution analysis
    print("\nError Distribution (percentiles):")
    sys.stdout.flush()
    percentiles = [50, 90, 95, 99, 99.9, 100]
    for p in percentiles:
        abs_val = torch.quantile(abs_diff, p / 100).item()
        rel_val = torch.quantile(rel_diff, p / 100).item()
        print(f"  {p:5.1f}%: abs={abs_val:.6e}, rel={rel_val:.6e}")

    # torch.allclose checks with different tolerances
    print("\nPassing torch.allclose() with different tolerances:")
    tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    for rtol in tolerances:
        passes = torch.allclose(pytorch_result, cuda_result, rtol=rtol, atol=1e-8)
        status = "✓ PASS" if passes else "✗ FAIL"
        print(f"  rtol={rtol:.0e}: {status}")

    # Value range analysis
    print("\nValue Range Analysis:")
    print(f"  PyTorch min:  {pytorch_result.min():.6e}")
    print(f"  PyTorch max:  {pytorch_result.max():.6e}")
    print(f"  PyTorch mean: {pytorch_result.mean():.6e}")
    print(f"  CUDA min:     {cuda_result.min():.6e}")
    print(f"  CUDA max:     {cuda_result.max():.6e}")
    print(f"  CUDA mean:    {cuda_result.mean():.6e}")

    # Check for NaN or Inf
    pytorch_nan = torch.isnan(pytorch_result).sum()
    cuda_nan = torch.isnan(cuda_result).sum()
    pytorch_inf = torch.isinf(pytorch_result).sum()
    cuda_inf = torch.isinf(cuda_result).sum()

    print("\nSpecial Values:")
    print(f"  PyTorch NaN:  {pytorch_nan.item()}")
    print(f"  PyTorch Inf:  {pytorch_inf.item()}")
    print(f"  CUDA NaN:     {cuda_nan.item()}")
    print(f"  CUDA Inf:     {cuda_inf.item()}")


def analyze_performance(size, pytorch_time, pytorch_std, cuda_time, cuda_std):
    """
    Detailed performance analysis including memory bandwidth.
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE ANALYSIS")
    print("=" * 70)

    # Basic timing
    print(f"\nTiming Results (avg ± std over 100 iterations):")
    print(f"  PyTorch (unfused):  {pytorch_time:.3f} ± {pytorch_std:.3f} ms")
    print(f"  Custom CUDA (fused): {cuda_time:.3f} ± {cuda_std:.3f} ms")

    speedup = pytorch_time / cuda_time
    print(f"\n  Speedup: {speedup:.2f}x")

    # Memory traffic analysis
    bytes_per_element = 4  # float32
    total_elements = size
    data_size_mb = (total_elements * bytes_per_element) / (1024**2)

    print(f"\nMemory Traffic Analysis:")
    print(f"  Data size: {total_elements:,} elements = {data_size_mb:.2f} MB")

    # PyTorch: read x, y, write a, read a, write b, read b, write c
    pytorch_traffic_mb = data_size_mb * 7  # 2 reads + 3 writes + 2 intermediate reads
    pytorch_bandwidth_gbs = (pytorch_traffic_mb / 1024) / (pytorch_time / 1000)

    print(f"\n  PyTorch (unfused):")
    print(f"    Memory operations: 7 (read x, y, a, b; write a, b, c)")
    print(f"    Total traffic:     {pytorch_traffic_mb:.2f} MB")
    print(f"    Bandwidth:         {pytorch_bandwidth_gbs:.2f} GB/s")

    # CUDA: read x, y, write c
    cuda_traffic_mb = data_size_mb * 3  # 2 reads + 1 write
    cuda_bandwidth_gbs = (cuda_traffic_mb / 1024) / (cuda_time / 1000)

    print(f"\n  Custom CUDA (fused):")
    print(f"    Memory operations: 3 (read x, y; write c)")
    print(f"    Total traffic:     {cuda_traffic_mb:.2f} MB")
    print(f"    Bandwidth:         {cuda_bandwidth_gbs:.2f} GB/s")

    # Theoretical analysis
    theoretical_speedup = pytorch_traffic_mb / cuda_traffic_mb
    efficiency = (speedup / theoretical_speedup) * 100

    print(f"\n  Efficiency Analysis:")
    print(f"    Theoretical max speedup: {theoretical_speedup:.2f}x")
    print(f"    Actual speedup:          {speedup:.2f}x")
    print(f"    Efficiency:              {efficiency:.1f}%")

    # GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"\n  GPU Information:")
        print(f"    Device:        {gpu_name}")
        print(f"    Total memory:  {gpu_memory_gb:.1f} GB")
        print(f"    Memory used:   {(data_size_mb * 3) / 1024:.2f} GB")
        print(f"                   (x, y, output tensors)")


def profile_with_pytorch_profiler(x, y):
    """
    Use PyTorch's profiler to get detailed kernel-level timing.
    """
    print("\n" + "=" * 70)
    print("PYTORCH PROFILER (Kernel-level timing)")
    print("=" * 70)

    # Profile PyTorch version
    print("\nProfiling PyTorch (unfused) - 10 iterations...")
    sys.stdout.flush()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=False,
    ) as prof:
        for _ in tqdm(range(10), desc="  ", leave=False, ncols=70):
            _ = pytorch_unfused(x, y)

    print("\nPyTorch (unfused) - Top 10 CUDA operations:")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10, max_src_column_width=50
        )
    )

    # Profile CUDA version
    print("\nProfiling Custom CUDA (fused) - 10 iterations...")
    sys.stdout.flush()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=False,
    ) as prof:
        for _ in tqdm(range(10), desc="  ", leave=False, ncols=70):
            _ = cuda_fused(x, y)

    print("\nCustom CUDA (fused) - Top 10 CUDA operations:")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10, max_src_column_width=50
        )
    )


def main():
    """
    Main test and analysis routine.
    """
    # Configuration
    size = 10_000_000  # 10 million elements (40 MB per tensor)

    # Create test data
    print(f"\n[SETUP] Test Configuration:")
    print(f"  Array size:     {size:,} elements")
    print(f"  Data type:      float32")
    print(f"  Memory/tensor:  {(size * 4) / (1024**2):.2f} MB")
    print(f"  Total memory:   {(size * 4 * 3) / (1024**2):.2f} MB (x, y, output)")

    print("\n[SETUP] Allocating tensors on GPU...")
    sys.stdout.flush()
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    print("✓ Tensors allocated")

    # Quick test to ensure CUDA kernel works
    print("\n[SETUP] Testing CUDA kernel...")
    sys.stdout.flush()
    try:
        test_result = add_mul_exp(x[:100], y[:100])
        print("✓ CUDA kernel working")
    except Exception as e:
        print(f"✗ CUDA kernel failed: {e}")
        return

    # Benchmark both versions
    print("\n" + "=" * 70)
    print("BENCHMARKING (100 iterations each)")
    print("=" * 70)

    print("\n[1/2] PyTorch (unfused) - 3 separate kernels")
    sys.stdout.flush()
    pytorch_result, pytorch_time, pytorch_std = benchmark(
        pytorch_unfused, x, y, "PyTorch (unfused)"
    )

    print("\n[2/2] Custom CUDA (fused) - 1 kernel")
    sys.stdout.flush()
    cuda_result, cuda_time, cuda_std = benchmark(
        cuda_fused, x, y, "Custom CUDA (fused)"
    )

    # Performance analysis
    analyze_performance(size, pytorch_time, pytorch_std, cuda_time, cuda_std)

    # Numerical accuracy analysis
    analyze_numerical_accuracy(pytorch_result, cuda_result)

    # Detailed profiling
    profile_with_pytorch_profiler(x, y)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    speedup = pytorch_time / cuda_time
    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Max abs error:    {torch.abs(pytorch_result - cuda_result).max():.6e}")

    # Get max relative error
    abs_diff = torch.abs(pytorch_result - cuda_result)
    denominator = torch.maximum(torch.abs(pytorch_result), torch.abs(cuda_result))
    denominator = torch.clamp(denominator, min=1e-10)
    rel_diff = abs_diff / denominator
    print(f"  Max rel error:    {rel_diff.max():.6e}")

    passes = torch.allclose(pytorch_result, cuda_result, rtol=1e-5, atol=1e-8)
    print(f"  Numerical match:  {'✓ PASS' if passes else '✗ FAIL'} (rtol=1e-5)")

    print("\n" + "=" * 70)
    print("Why is custom CUDA faster?")
    print("  - PyTorch:     7 memory operations (read x,y,a,b; write a,b,c)")
    print("  - Custom CUDA: 3 memory operations (read x,y; write c)")
    print("  - Intermediate values (a, b) stay in GPU registers (400x faster!)")
    print("  - Modern GPUs are memory-bound → less memory traffic = faster")
    print("=" * 70)


if __name__ == "__main__":
    main()

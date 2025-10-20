"""
Performance benchmarking utilities for CUDA kernels.

Provides reusable functions for timing, comparing, and profiling CUDA operations.
"""

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


@dataclass
class BenchmarkResult:
    """Results from benchmarking a function."""

    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    result: torch.Tensor
    iterations: int

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Mean:   {self.mean_time_ms:.3f} ± {self.std_time_ms:.3f} ms\n"
            f"  Median: {self.median_time_ms:.3f} ms\n"
            f"  Range:  [{self.min_time_ms:.3f}, {self.max_time_ms:.3f}] ms"
        )


def benchmark_function(
    func: Callable,
    args: Tuple,
    name: str = "Function",
    warmup: int = 10,
    iterations: int = 100,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Benchmark a function with warmup and multiple iterations.

    Args:
        func: Function to benchmark
        args: Tuple of arguments to pass to the function
        name: Name of the function for display
        warmup: Number of warmup iterations
        iterations: Number of timing iterations
        verbose: Whether to show progress bars

    Returns:
        BenchmarkResult with timing statistics and output

    Example:
        >>> x = torch.randn(1000, device='cuda')
        >>> y = torch.randn(1000, device='cuda')
        >>> result = benchmark_function(
        ...     lambda x, y: x + y,
        ...     (x, y),
        ...     name="Addition",
        ...     iterations=100
        ... )
        >>> print(result)
    """
    # Warmup
    if verbose:
        for _ in tqdm(range(warmup), desc=f"  {name} warmup", leave=False, ncols=70):
            _ = func(*args)
        torch.cuda.synchronize()

    # Collect timings
    times = []
    result = None

    iterator = (
        tqdm(range(iterations), desc=f"  {name} timing", leave=False, ncols=70)
        if verbose
        else range(iterations)
    )

    for _ in iterator:
        start = time.perf_counter()
        result = func(*args)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    # Convert to milliseconds
    times_ms = np.array(times) * 1000

    return BenchmarkResult(
        name=name,
        mean_time_ms=float(times_ms.mean()),
        std_time_ms=float(times_ms.std()),
        min_time_ms=float(times_ms.min()),
        max_time_ms=float(times_ms.max()),
        median_time_ms=float(np.median(times_ms)),
        result=result,
        iterations=iterations,
    )


def compare_implementations(
    baseline_func: Callable,
    optimized_func: Callable,
    args: Tuple,
    baseline_name: str = "Baseline",
    optimized_name: str = "Optimized",
    warmup: int = 10,
    iterations: int = 100,
    verbose: bool = True,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Compare two implementations and compute speedup.

    Args:
        baseline_func: Baseline implementation (e.g., PyTorch)
        optimized_func: Optimized implementation (e.g., custom CUDA)
        args: Tuple of arguments to pass to both functions
        baseline_name: Name for baseline implementation
        optimized_name: Name for optimized implementation
        warmup: Number of warmup iterations
        iterations: Number of timing iterations
        verbose: Whether to show progress and results

    Returns:
        Tuple of (baseline_result, optimized_result)

    Example:
        >>> def pytorch_impl(x, y):
        ...     return torch.exp((x + y) * 2)
        >>> def cuda_impl(x, y):
        ...     return add_mul_exp(x, y)
        >>> x = torch.randn(1000000, device='cuda')
        >>> y = torch.randn(1000000, device='cuda')
        >>> baseline, optimized = compare_implementations(
        ...     pytorch_impl, cuda_impl, (x, y)
        ... )
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print("BENCHMARKING")
        print(f"{'=' * 70}")

    # Benchmark baseline
    if verbose:
        print(f"\n[1/2] {baseline_name}")
    baseline_result = benchmark_function(
        baseline_func,
        args,
        name=baseline_name,
        warmup=warmup,
        iterations=iterations,
        verbose=verbose,
    )

    # Benchmark optimized
    if verbose:
        print(f"\n[2/2] {optimized_name}")
    optimized_result = benchmark_function(
        optimized_func,
        args,
        name=optimized_name,
        warmup=warmup,
        iterations=iterations,
        verbose=verbose,
    )

    # Display results
    if verbose:
        print(f"\n{'=' * 70}")
        print("TIMING RESULTS")
        print(f"{'=' * 70}\n")
        print(baseline_result)
        print()
        print(optimized_result)

        speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms
        print(f"\n  Speedup: {speedup:.2f}x")

    return baseline_result, optimized_result


def analyze_memory_traffic(
    tensor_size: int,
    baseline_reads: int,
    baseline_writes: int,
    optimized_reads: int,
    optimized_writes: int,
    baseline_time_ms: float,
    optimized_time_ms: float,
    dtype: torch.dtype = torch.float32,
    baseline_kernel_launches: int = 1,
    optimized_kernel_launches: int = 1,
) -> dict:
    """
    Analyze memory traffic and bandwidth for two implementations.

    Args:
        tensor_size: Number of elements in tensors
        baseline_reads: Number of tensor reads in baseline
        baseline_writes: Number of tensor writes in baseline
        optimized_reads: Number of tensor reads in optimized version
        optimized_writes: Number of tensor writes in optimized version
        baseline_time_ms: Baseline execution time in milliseconds
        optimized_time_ms: Optimized execution time in milliseconds
        dtype: Data type of tensors (default: float32)
        baseline_kernel_launches: Number of kernel launches in baseline (default: 1)
        optimized_kernel_launches: Number of kernel launches in optimized (default: 1)

    Returns:
        Dictionary with memory traffic analysis

    Example:
        >>> stats = analyze_memory_traffic(
        ...     tensor_size=10_000_000,
        ...     baseline_reads=4,      # PyTorch: read x, y, a, b
        ...     baseline_writes=3,     # PyTorch: write a, b, c
        ...     optimized_reads=2,     # CUDA: read x, y
        ...     optimized_writes=1,    # CUDA: write c
        ...     baseline_time_ms=0.686,
        ...     optimized_time_ms=0.297,
        ...     baseline_kernel_launches=3,
        ...     optimized_kernel_launches=1
        ... )
    """
    bytes_per_element = torch.finfo(dtype).bits // 8

    # Calculate data size
    data_size_mb = (tensor_size * bytes_per_element) / (1024**2)

    # Calculate traffic
    baseline_traffic_mb = data_size_mb * (baseline_reads + baseline_writes)
    optimized_traffic_mb = data_size_mb * (optimized_reads + optimized_writes)

    # Calculate bandwidth (MB/s -> GB/s)
    baseline_bandwidth_gbs = (baseline_traffic_mb / 1024) / (baseline_time_ms / 1000)
    optimized_bandwidth_gbs = (optimized_traffic_mb / 1024) / (optimized_time_ms / 1000)

    # Calculate theoretical speedup using a more realistic model
    #
    # Naive model: speedup = memory_traffic_reduction = baseline_ops / optimized_ops
    # This misses several real-world factors:
    #
    # 1. Kernel launch overhead (~5-10μs per launch)
    #    - PyTorch: N launches, CUDA: 1 launch
    #    - Savings: (N-1) * 7.5μs
    #
    # 2. Cache/Register locality
    #    - Fused kernels keep intermediates in registers/L1 cache
    #    - Unfused operations read/write through slow DRAM
    #    - L1 cache is ~10-20x faster than DRAM
    #
    # 3. Reduced synchronization overhead
    #    - PyTorch implicitly syncs between operations
    #    - Fused kernel runs continuously

    # Base speedup from memory traffic reduction
    memory_speedup = baseline_traffic_mb / optimized_traffic_mb

    # Kernel launch overhead contribution
    # Estimate: 10μs per launch (conservative)
    KERNEL_LAUNCH_US = 10
    baseline_launch_overhead_us = baseline_kernel_launches * KERNEL_LAUNCH_US
    optimized_launch_overhead_us = optimized_kernel_launches * KERNEL_LAUNCH_US
    launch_overhead_saved_us = (
        baseline_launch_overhead_us - optimized_launch_overhead_us
    )

    # Convert execution time to microseconds
    baseline_time_us = baseline_time_ms * 1000

    # Launch overhead as fraction of total time
    launch_overhead_fraction = launch_overhead_saved_us / baseline_time_us
    launch_speedup_contribution = (
        1.0 / (1.0 - launch_overhead_fraction)
        if launch_overhead_fraction < 1.0
        else 1.0
    )

    # Cache locality benefit (empirical factor)
    # Fused kernels keep ~50-80% of intermediate data in registers/L1
    # This provides 1.2-1.5x additional speedup beyond memory traffic reduction
    cache_locality_factor = 1.5  # Conservative estimate

    # Combined theoretical speedup
    theoretical_speedup = (
        memory_speedup * launch_speedup_contribution * cache_locality_factor
    )

    actual_speedup = baseline_time_ms / optimized_time_ms
    efficiency = (actual_speedup / theoretical_speedup) * 100

    return {
        "tensor_size": tensor_size,
        "data_size_mb": data_size_mb,
        "baseline": {
            "reads": baseline_reads,
            "writes": baseline_writes,
            "total_ops": baseline_reads + baseline_writes,
            "traffic_mb": baseline_traffic_mb,
            "bandwidth_gbs": baseline_bandwidth_gbs,
            "time_ms": baseline_time_ms,
            "kernel_launches": baseline_kernel_launches,
        },
        "optimized": {
            "reads": optimized_reads,
            "writes": optimized_writes,
            "total_ops": optimized_reads + optimized_writes,
            "traffic_mb": optimized_traffic_mb,
            "bandwidth_gbs": optimized_bandwidth_gbs,
            "time_ms": optimized_time_ms,
            "kernel_launches": optimized_kernel_launches,
        },
        "speedup": {
            "theoretical": theoretical_speedup,
            "actual": actual_speedup,
            "efficiency_pct": efficiency,
            "memory_traffic_reduction": memory_speedup,
        },
    }


def print_memory_analysis(analysis: dict):
    """
    Pretty print memory traffic analysis.

    Args:
        analysis: Dictionary from analyze_memory_traffic()
    """
    print(f"\n{'=' * 70}")
    print("MEMORY TRAFFIC ANALYSIS")
    print(f"{'=' * 70}")

    print(
        f"\nData size: {analysis['tensor_size']:,} elements = {analysis['data_size_mb']:.2f} MB"
    )

    print(f"\nBaseline:")
    print(f"  Kernel launches:   {analysis['baseline']['kernel_launches']}")
    print(f"  Memory operations: {analysis['baseline']['total_ops']} ")
    print(
        f"    ({analysis['baseline']['reads']} reads + {analysis['baseline']['writes']} writes)"
    )
    print(f"  Total traffic:     {analysis['baseline']['traffic_mb']:.2f} MB")
    print(f"  Bandwidth:         {analysis['baseline']['bandwidth_gbs']:.2f} GB/s")

    print(f"\nOptimized:")
    print(f"  Kernel launches:   {analysis['optimized']['kernel_launches']}")
    print(f"  Memory operations: {analysis['optimized']['total_ops']} ")
    print(
        f"    ({analysis['optimized']['reads']} reads + {analysis['optimized']['writes']} writes)"
    )
    print(f"  Total traffic:     {analysis['optimized']['traffic_mb']:.2f} MB")
    print(f"  Bandwidth:         {analysis['optimized']['bandwidth_gbs']:.2f} GB/s")

    print(f"\nSpeedup Analysis:")
    print(
        f"  Memory reduction:       {analysis['speedup']['memory_traffic_reduction']:.2f}x"
    )
    print(
        f"  Kernel reduction:       {analysis['baseline']['kernel_launches']}/{analysis['optimized']['kernel_launches']} = {analysis['baseline']['kernel_launches']/analysis['optimized']['kernel_launches']:.2f}x"
    )
    print(f"  Theoretical speedup:    {analysis['speedup']['theoretical']:.2f}x")
    print(f"  Actual speedup:         {analysis['speedup']['actual']:.2f}x")
    print(f"  Efficiency:             {analysis['speedup']['efficiency_pct']:.1f}%")


def profile_with_pytorch_profiler(
    funcs_and_names: List[Tuple[Callable, str, Tuple]],
    iterations: int = 10,
) -> None:
    """
    Profile multiple functions using PyTorch's profiler.

    Args:
        funcs_and_names: List of (function, name, args) tuples to profile
        iterations: Number of iterations to profile

    Example:
        >>> profile_with_pytorch_profiler([
        ...     (pytorch_impl, "PyTorch", (x, y)),
        ...     (cuda_impl, "CUDA", (x, y))
        ... ])
    """
    print(f"\n{'=' * 70}")
    print("PYTORCH PROFILER (Kernel-level timing)")
    print(f"{'=' * 70}")

    for func, name, args in funcs_and_names:
        print(f"\nProfiling {name} - {iterations} iterations...")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_stack=False,
        ) as prof:
            for _ in tqdm(range(iterations), desc="  ", leave=False, ncols=70):
                _ = func(*args)

        print(f"\n{name} - Top 10 CUDA operations:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

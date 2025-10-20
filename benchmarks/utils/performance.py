"""
Performance benchmarking utilities for CUDA kernels.

Provides reusable functions for timing, comparing, and profiling CUDA operations.
"""

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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


def measure_gpu_specs(size_mb: int = 1000, num_iterations: int = 100, verbose: bool = True) -> dict:
    """
    Measure actual achievable GPU memory bandwidth and estimate peak FLOPs.

    This provides more accurate specs than manufacturer specifications, as it
    measures what's actually achievable on your specific hardware configuration.

    Args:
        size_mb: Size of data to transfer in MB (default: 1000 MB)
        num_iterations: Number of iterations for measurement (default: 100)
        verbose: Print measurement progress

    Returns:
        Dictionary with 'peak_bandwidth' (bytes/sec) and 'peak_flops' (FLOPs/sec)

    Example:
        >>> specs = measure_gpu_specs()
        >>> print(f"Bandwidth: {specs['peak_bandwidth']/1e9:.1f} GB/s")
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    if verbose:
        print("\n[Measuring GPU specs...]")

    size = size_mb * 1024 * 1024 // 4  # Convert MB to float32 elements

    # Measure memory bandwidth
    if verbose:
        print(f"  Measuring memory bandwidth ({size_mb} MB, {num_iterations} iterations)...")

    src = torch.randn(size, device='cuda', dtype=torch.float32)
    dst = torch.empty_like(src)

    # Warmup
    for _ in range(10):
        dst.copy_(src)
    torch.cuda.synchronize()

    # Measure bandwidth
    start = time.perf_counter()
    for _ in range(num_iterations):
        dst.copy_(src)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Calculate bandwidth (read + write)
    bytes_transferred = size * 4 * 2 * num_iterations  # float32 = 4 bytes, read+write
    measured_bandwidth = bytes_transferred / elapsed

    if verbose:
        print(f"  ✓ Measured bandwidth: {measured_bandwidth/1e9:.1f} GB/s")

    # For FLOPs, use a simple estimation based on GPU architecture
    # This is a rough estimate - actual peak depends on specific operations
    # For now, we'll use manufacturer specs but this could be improved with
    # actual FLOP measurement using FMAD operations
    gpu_name = torch.cuda.get_device_name(0)

    # Common GPU specs (FP32 TFLOPS)
    gpu_flops_specs = {
        "RTX 3070": 20.3e12,
        "RTX 3080": 29.8e12,
        "RTX 3090": 35.6e12,
        "RTX 4090": 82.6e12,
        "A100": 19.5e12,
        "V100": 15.7e12,
    }

    # Try to match GPU name
    estimated_flops = None
    for gpu_model, flops in gpu_flops_specs.items():
        if gpu_model in gpu_name:
            estimated_flops = flops
            break

    if estimated_flops is None:
        # Default conservative estimate: 10 TFLOPS
        estimated_flops = 10e12
        if verbose:
            print(f"  ! Unknown GPU, using conservative FLOP estimate: {estimated_flops/1e12:.1f} TFLOPS")
    else:
        if verbose:
            print(f"  ✓ Estimated peak FLOPs: {estimated_flops/1e12:.1f} TFLOPS")

    return {
        'peak_bandwidth': measured_bandwidth,
        'peak_flops': estimated_flops,
    }


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


def roofline_efficiency(kernel_stats: dict, gpu_specs: dict) -> tuple:
    """
    Calculate efficiency with proper handling of cache effects.

    Uses roofline model to determine theoretical best time, then compares to actual.
    When actual time is better than theoretical, cache effects are detected and
    reported separately.

    Args:
        kernel_stats: dict with 'flops', 'bytes', 'time_ms'
        gpu_specs: dict with 'peak_flops', 'peak_bandwidth'

    Returns:
        efficiency: 0-1 (0-100%), capped at 100%
        bottleneck: 'compute' or 'memory'
        cache_benefit: >= 1.0, indicates how much cache is helping

    Example:
        >>> stats = {
        ...     'flops': 10_000_000,  # Total floating point operations
        ...     'bytes': 40_000_000,  # Total bytes transferred
        ...     'time_ms': 0.5        # Execution time in ms
        ... }
        >>> specs = {
        ...     'peak_flops': 20e12,      # 20 TFLOPS
        ...     'peak_bandwidth': 448e9   # 448 GB/s
        ... }
        >>> efficiency, bottleneck, cache_benefit = roofline_efficiency(stats, specs)
    """
    # Calculate arithmetic intensity (FLOPs per byte)
    arithmetic_intensity = kernel_stats["flops"] / kernel_stats["bytes"]

    # Ridge point: where compute and memory bounds intersect
    ridge_point = gpu_specs["peak_flops"] / gpu_specs["peak_bandwidth"]

    # Calculate minimum achievable time based on roofline model
    # Time is limited by max(compute_time, memory_time)
    compute_time_s = kernel_stats["flops"] / gpu_specs["peak_flops"]
    memory_time_s = kernel_stats["bytes"] / gpu_specs["peak_bandwidth"]

    if arithmetic_intensity < ridge_point:
        # Memory-bound: memory transfer is the bottleneck
        theoretical_time_s = memory_time_s
        bottleneck = "memory"
    else:
        # Compute-bound: computation is the bottleneck
        theoretical_time_s = compute_time_s
        bottleneck = "compute"

    actual_time_s = kernel_stats["time_ms"] * 1e-3

    # Calculate raw efficiency ratio
    # raw_efficiency > 1.0 means actual is faster than theoretical (cache helping!)
    # raw_efficiency < 1.0 means actual is slower than theoretical (inefficient)
    raw_efficiency = theoretical_time_s / actual_time_s

    # Cap efficiency at 100% - can't be more efficient than theoretically perfect
    efficiency = min(1.0, raw_efficiency)

    # Cache benefit: how much faster than theoretical prediction
    # >1.0 means cache/optimizations are helping beyond our simple model
    # If raw_efficiency = 1.5, we're 1.5x faster than theory, so cache_benefit = 1.5x
    cache_benefit = raw_efficiency if raw_efficiency > 1.0 else 1.0

    return efficiency, bottleneck, cache_benefit


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
    baseline_flops: Optional[int] = None,
    optimized_flops: Optional[int] = None,
    gpu_specs: Optional[dict] = None,
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
        baseline_flops: Optional FLOPs for baseline (enables roofline analysis)
        optimized_flops: Optional FLOPs for optimized (enables roofline analysis)
        gpu_specs: Optional GPU specs dict with 'peak_flops' and 'peak_bandwidth'
                   (enables roofline analysis)

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

    # Calculate speedup
    actual_speedup = baseline_time_ms / optimized_time_ms
    memory_speedup = baseline_traffic_mb / optimized_traffic_mb

    # Use roofline model if FLOP counts and GPU specs are provided
    use_roofline = (
        baseline_flops is not None
        and optimized_flops is not None
        and gpu_specs is not None
    )

    if use_roofline:
        # Roofline-based efficiency analysis
        baseline_stats = {
            "flops": baseline_flops,
            "bytes": baseline_traffic_mb * 1024 * 1024,  # Convert to bytes
            "time_ms": baseline_time_ms,
        }
        optimized_stats = {
            "flops": optimized_flops,
            "bytes": optimized_traffic_mb * 1024 * 1024,  # Convert to bytes
            "time_ms": optimized_time_ms,
        }

        baseline_efficiency, baseline_bottleneck, baseline_cache_benefit = roofline_efficiency(
            baseline_stats, gpu_specs
        )
        optimized_efficiency, optimized_bottleneck, optimized_cache_benefit = roofline_efficiency(
            optimized_stats, gpu_specs
        )

        # Detect significant cache effects (>10% faster than theoretical)
        baseline_cache_detected = baseline_cache_benefit > 1.1
        optimized_cache_detected = optimized_cache_benefit > 1.1

        speedup_analysis = {
            "actual": actual_speedup,
            "memory_traffic_reduction": memory_speedup,
            "baseline_efficiency_pct": baseline_efficiency * 100,
            "optimized_efficiency_pct": optimized_efficiency * 100,
            "baseline_cache_benefit": baseline_cache_benefit,
            "optimized_cache_benefit": optimized_cache_benefit,
            "baseline_cache_detected": baseline_cache_detected,
            "optimized_cache_detected": optimized_cache_detected,
            "baseline_bottleneck": baseline_bottleneck,
            "optimized_bottleneck": optimized_bottleneck,
            "baseline_arithmetic_intensity": baseline_flops
            / (baseline_traffic_mb * 1024 * 1024),
            "optimized_arithmetic_intensity": optimized_flops
            / (optimized_traffic_mb * 1024 * 1024),
        }
    else:
        # No efficiency calculation without roofline data
        # Simple memory reduction is reported but no efficiency metric
        speedup_analysis = {
            "actual": actual_speedup,
            "memory_traffic_reduction": memory_speedup,
        }

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
        "speedup": speedup_analysis,
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

    print(f"  Actual speedup:         {analysis['speedup']['actual']:.2f}x")

    # Check if using roofline model
    if "baseline_bottleneck" in analysis["speedup"]:
        # Roofline-based efficiency analysis
        print(f"\nRoofline Analysis:")
        print(
            f"  Baseline arithmetic intensity:  {analysis['speedup']['baseline_arithmetic_intensity']:.3f} FLOPs/byte"
        )
        print(
            f"  Baseline bottleneck:            {analysis['speedup']['baseline_bottleneck']}"
        )

        # Show efficiency and cache benefit for baseline
        print(
            f"  Baseline efficiency:            {analysis['speedup']['baseline_efficiency_pct']:.1f}%"
        )
        baseline_cache = analysis['speedup'].get('baseline_cache_detected', False)
        if baseline_cache:
            baseline_benefit = analysis['speedup']['baseline_cache_benefit']
            print(
                f"  Baseline cache benefit:         {baseline_benefit:.2f}x ✓ (faster than theory predicts)"
            )

        print(
            f"  Optimized arithmetic intensity: {analysis['speedup']['optimized_arithmetic_intensity']:.3f} FLOPs/byte"
        )
        print(
            f"  Optimized bottleneck:           {analysis['speedup']['optimized_bottleneck']}"
        )
        print(
            f"  Optimized efficiency:           {analysis['speedup']['optimized_efficiency_pct']:.1f}%"
        )

        # Show cache benefit for optimized
        optimized_cache = analysis['speedup'].get('optimized_cache_detected', False)
        if optimized_cache:
            optimized_benefit = analysis['speedup']['optimized_cache_benefit']
            print(
                f"  Optimized cache benefit:        {optimized_benefit:.2f}x ✓ (faster than theory predicts)"
            )

        # Show explanation if cache detected
        if baseline_cache or optimized_cache:
            print(f"\n  ⚠️  Cache effects detected - kernel benefits from L2 cache (GOOD!)")
            print(f"      Cache benefit >1.0x means actual performance exceeds theoretical prediction")
            print(f"      This indicates excellent data locality and cache utilization")


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

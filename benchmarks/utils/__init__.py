"""
Benchmarking utilities for CUDA kernel fusion.
"""

from .analysis import AccuracyMetrics, analyze_numerical_accuracy
from .performance import (
    BenchmarkResult,
    analyze_memory_traffic,
    benchmark_function,
    compare_three_implementations,
    measure_gpu_specs,
    print_memory_analysis,
    profile_with_pytorch_profiler,
    roofline_efficiency,
)

__all__ = [
    # Performance
    "BenchmarkResult",
    "benchmark_function",
    "compare_three_implementations",
    "measure_gpu_specs",
    "roofline_efficiency",
    "analyze_memory_traffic",
    "print_memory_analysis",
    "profile_with_pytorch_profiler",
    # Analysis
    "AccuracyMetrics",
    "analyze_numerical_accuracy",
]

"""
Benchmarking utilities for CUDA kernel fusion.
"""

from .performance import (
    benchmark_function,
    compare_implementations,
    BenchmarkResult,
    analyze_memory_traffic,
    print_memory_analysis,
    profile_with_pytorch_profiler,
)
from .analysis import (
    analyze_numerical_accuracy,
    AccuracyMetrics,
    compare_accuracy_quick,
    assert_accuracy,
)

__all__ = [
    # Performance
    "benchmark_function",
    "compare_implementations",
    "BenchmarkResult",
    "analyze_memory_traffic",
    "print_memory_analysis",
    "profile_with_pytorch_profiler",
    # Analysis
    "analyze_numerical_accuracy",
    "AccuracyMetrics",
    "compare_accuracy_quick",
    "assert_accuracy",
]

"""
Benchmark for INT8 quantization kernel.

Compares fused CUDA implementation against PyTorch operations.
Operation: quantize_int8(x, scale, zero_point) -> int8
"""

import os
import sys

# Set CUDA architecture BEFORE importing torch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch
from utils.analysis import analyze_numerical_accuracy
from utils.performance import (
    analyze_memory_traffic,
    compare_implementations,
    measure_gpu_specs,
    print_memory_analysis,
    profile_with_pytorch_profiler,
)

# ============================================================================
# BENCHMARK CONFIGURATION - Modify this section for new kernels
# ============================================================================
OPERATION_NAME = "INT8 Quantization"
OPERATION_DESCRIPTION = "FP32 -> INT8 quantization with scale and zero_point"

# Display header
print("=" * 70)
print(f"CUDA KERNEL FUSION BENCHMARK: {OPERATION_NAME}")
print("=" * 70)
print(f"\nOperation: {OPERATION_DESCRIPTION}")
print("Comparing: PyTorch vs Custom CUDA")

# Load CUDA extension
print("\n[SETUP] Loading CUDA extension...")
sys.stdout.flush()

try:
    from ops.cuda import quantize_int8_cuda
    from ops.torch.quantize_int8 import quantize_int8_pytorch

    # Generic function references
    baseline_func = quantize_int8_pytorch
    optimized_func = quantize_int8_cuda

    print("✓ CUDA extension loaded")
except ImportError as e:
    print(f"✗ Failed to import ops: {e}")
    print("\nPlease install the package first:")
    print("  pip install --no-build-isolation -e .")
    sys.exit(1)


def main():
    """Run comprehensive benchmark."""
    # ========================================================================
    # BENCHMARK PARAMETERS - Modify for your operation
    # ========================================================================
    size = 10_000_000
    warmup_iterations = 10
    benchmark_iterations = 100
    profiler_iterations = 10

    # Memory operation counts and FLOPs for roofline analysis
    # PyTorch: div(x, scale) -> add(a, zp) -> round(b) -> clamp(c, -128, 127) -> to_int8(d)
    #   Reads: x, a, b, c, d (5 reads from DRAM)
    #   Writes: a, b, c, d, result (5 writes to DRAM)
    #   FLOPs: 4 ops per element (div, add, round, clamp) = 4N
    # CUDA: All operations fused, intermediates stay in registers
    #   Reads: x (1 read from DRAM)
    #   Writes: result (1 write to DRAM)
    #   FLOPs: 4 ops per element (div, add, round, clamp) = 4N
    memory_config = {
        "baseline_reads": 5,
        "baseline_writes": 5,
        "baseline_kernel_launches": 5,
        "optimized_reads": 1,
        "optimized_writes": 1,
        "optimized_kernel_launches": 1,
        "baseline_flops": 4 * size,  # div + add + round + clamp
        "optimized_flops": 4 * size,  # div + add + round + clamp
    }

    # ========================================================================
    # MEASURE GPU SPECS
    # ========================================================================
    # Measure actual achievable bandwidth instead of using theoretical specs
    gpu_specs = measure_gpu_specs(verbose=True)

    # ========================================================================
    # SETUP
    # ========================================================================
    print(f"\n[SETUP] Test Configuration:")
    print(f"  Array size:     {size:,} elements")
    print(f"  Data type:      float32")
    print(f"  Memory/tensor:  {(size * 4) / (1024**2):.2f} MB")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n✗ CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU:            {gpu_name}")

    # Prepare benchmark arguments
    print("\n[SETUP] Allocating tensors...")
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    scale = 128.0
    zero_point = 0.0
    benchmark_args = (x, scale, zero_point)
    print("✓ Tensors allocated")

    # Quick sanity check
    print("\n[SETUP] Testing CUDA kernel...")
    try:
        _ = optimized_func(x[:100], scale, zero_point)
        print("✓ CUDA kernel working")
    except Exception as e:
        print(f"✗ CUDA kernel failed: {e}")
        sys.exit(1)

    # ========================================================================
    # BENCHMARKING
    # ========================================================================
    baseline_result, optimized_result = compare_implementations(
        baseline_func=baseline_func,
        optimized_func=optimized_func,
        args=benchmark_args,
        baseline_name="PyTorch",
        optimized_name="CUDA",
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        verbose=True,
    )

    # ========================================================================
    # MEMORY TRAFFIC ANALYSIS
    # ========================================================================
    memory_stats = analyze_memory_traffic(
        tensor_size=size,
        baseline_time_ms=baseline_result.mean_time_ms,
        optimized_time_ms=optimized_result.mean_time_ms,
        gpu_specs=gpu_specs,
        **memory_config,
    )
    print_memory_analysis(memory_stats)

    # ========================================================================
    # NUMERICAL ACCURACY ANALYSIS
    # ========================================================================
    accuracy_metrics = analyze_numerical_accuracy(
        baseline=baseline_result.result,
        optimized=optimized_result.result,
        verbose=True,
    )

    # ========================================================================
    # PYTORCH PROFILER (Kernel-level timing)
    # ========================================================================
    profile_with_pytorch_profiler(
        [
            (baseline_func, "PyTorch", benchmark_args),
            (optimized_func, "CUDA", benchmark_args),
        ],
        iterations=profiler_iterations,
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms
    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Max abs error:    {accuracy_metrics.max_abs_diff:.6e}")
    print(f"  Max rel error:    {accuracy_metrics.max_rel_diff:.6e}")
    print(
        f"  Passes rtol=1e-5: {'✓ PASS' if accuracy_metrics.passes_1e5 else '✗ FAIL'}"
    )


if __name__ == "__main__":
    main()

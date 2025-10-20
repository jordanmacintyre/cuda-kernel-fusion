"""
Benchmark template for new CUDA kernels.

Copy this file and modify the CONFIGURATION sections for your custom operation.
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
    print_memory_analysis,
    profile_with_pytorch_profiler,
)

# ============================================================================
# BENCHMARK CONFIGURATION - Modify this section for new kernels
# ============================================================================
OPERATION_NAME = "<YOUR_OPERATION>"  # e.g., "Add-Mul-Exp Fusion"
OPERATION_DESCRIPTION = "<DESCRIBE_OPERATION>"  # e.g., "exp((x + y) * 2)"

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
    # TODO: Import your custom operations
    # from ops.cuda import your_operation_cuda
    # from ops.torch import your_operation_pytorch
    raise NotImplementedError("Replace with your imports")

    # Generic function references (uncomment after importing)
    # baseline_func = your_operation_pytorch
    # optimized_func = your_operation_cuda

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

    # Memory operation counts (for theoretical speedup analysis)
    # TODO: Count memory operations for your kernel
    # Example:
    #   PyTorch: operation1(x, y) -> operation2(a) -> operation3(b)
    #     Reads: x, y, a, b (4 reads from DRAM)
    #     Writes: a, b, result (3 writes to DRAM)
    #   CUDA: All operations fused, intermediates stay in registers
    #     Reads: x, y (2 reads from DRAM)
    #     Writes: result (1 write to DRAM)
    memory_config = {
        "baseline_reads": 4,  # TODO: Count baseline reads
        "baseline_writes": 3,  # TODO: Count baseline writes
        "baseline_kernel_launches": 3,  # TODO: Count baseline kernel launches
        "optimized_reads": 2,  # TODO: Count optimized reads
        "optimized_writes": 1,  # TODO: Count optimized writes
        "optimized_kernel_launches": 1,  # Usually 1 for fused kernels
    }

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
    # TODO: Allocate your input tensors
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    # Add any additional parameters your operation needs
    benchmark_args = (x, y)  # TODO: Update with your operation's arguments
    print("✓ Tensors allocated")

    # Quick sanity check
    print("\n[SETUP] Testing CUDA kernel...")
    try:
        # TODO: Test your CUDA kernel with small input
        # _ = optimized_func(x[:100], y[:100])
        raise NotImplementedError("Test your CUDA kernel here")
        print("✓ CUDA kernel working")
    except Exception as e:
        print(f"✗ CUDA kernel failed: {e}")
        sys.exit(1)

    # ========================================================================
    # BENCHMARKING
    # ========================================================================
    baseline_result, optimized_result = compare_implementations(
        baseline_func=baseline_func,  # Set in CONFIGURATION section
        optimized_func=optimized_func,  # Set in CONFIGURATION section
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

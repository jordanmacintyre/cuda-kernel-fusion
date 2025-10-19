"""
Benchmark template for new CUDA kernels.

Copy this file and modify for your custom operation.
"""

import os
import sys

# Set CUDA architecture BEFORE importing torch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import torch

from utils import (
    analyze_numerical_accuracy,
    analyze_memory_traffic,
    compare_implementations,
    print_memory_analysis,
)

print("=" * 70)
print("CUDA KERNEL FUSION BENCHMARK: <YOUR_OPERATION>")
print("=" * 70)
print("\nOperation: <describe your operation>")
print("Comparing: PyTorch vs Custom CUDA")

# Load CUDA extension
print("\n[SETUP] Loading CUDA extension...")
sys.stdout.flush()

try:
    # TODO: Import your custom operation using _cuda and _pytorch suffixes
    # from ops.cuda import your_operation_cuda
    # from ops.torch import your_operation_pytorch
    raise NotImplementedError("Replace this with your imports")
except ImportError as e:
    print(f"✗ Failed to import ops: {e}")
    print("\nPlease install the package first:")
    print("  pip install --no-build-isolation -e .")
    sys.exit(1)

# NOTE: Naming convention:
# - your_operation_cuda(): CUDA implementation (in ops/cuda/your_op.py)
# - your_operation_pytorch(): PyTorch baseline (in ops/torch/your_op.py)
# This keeps code organized and follows industry standards


def main():
    """Run comprehensive benchmark."""
    # Configuration
    size = 10_000_000  # Adjust as needed

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

    # Allocate tensors
    print("\n[SETUP] Allocating tensors...")
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    print("✓ Tensors allocated")

    # Quick sanity check
    print("\n[SETUP] Testing CUDA kernel...")
    try:
        # TODO: Replace with your CUDA operation
        # _ = your_operation_cuda(x[:100], y[:100])
        raise NotImplementedError("Test your CUDA kernel here")
        print("✓ CUDA kernel working")
    except Exception as e:
        print(f"✗ CUDA kernel failed: {e}")
        sys.exit(1)

    # Benchmark both implementations
    baseline_result, optimized_result = compare_implementations(
        baseline_func=None,  # TODO: your_operation_pytorch,
        optimized_func=None,  # TODO: your_operation_cuda,
        args=(x, y),
        baseline_name="PyTorch",
        optimized_name="CUDA",
        warmup=10,
        iterations=100,
        verbose=True,
    )

    # TODO: Update memory operation counts for your kernel
    memory_stats = analyze_memory_traffic(
        tensor_size=size,
        baseline_reads=4,  # TODO: Count PyTorch reads
        baseline_writes=3,  # TODO: Count PyTorch writes
        optimized_reads=2,  # TODO: Count CUDA reads
        optimized_writes=1,  # TODO: Count CUDA writes
        baseline_time_ms=baseline_result.mean_time_ms,
        optimized_time_ms=optimized_result.mean_time_ms,
    )
    print_memory_analysis(memory_stats)

    # Numerical accuracy analysis
    accuracy_metrics = analyze_numerical_accuracy(
        baseline=baseline_result.result,
        optimized=optimized_result.result,
        verbose=True,
    )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    speedup = baseline_result.mean_time_ms / optimized_result.mean_time_ms
    print(f"  Speedup:          {speedup:.2f}x")
    print(f"  Max abs error:    {accuracy_metrics.max_abs_diff:.6e}")
    print(f"  Max rel error:    {accuracy_metrics.max_rel_diff:.6e}")
    print(f"  Passes rtol=1e-5: {'✓ PASS' if accuracy_metrics.passes_1e5 else '✗ FAIL'}")


if __name__ == "__main__":
    main()

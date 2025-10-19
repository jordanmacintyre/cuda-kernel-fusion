"""
Benchmark for add_mul_exp kernel fusion.

Compares fused CUDA implementation against unfused PyTorch operations.
Operation: exp((x + y) * 2)
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
    profile_with_pytorch_profiler,
)

print("=" * 70)
print("CUDA KERNEL FUSION BENCHMARK: add_mul_exp")
print("=" * 70)
print("\nOperation: exp((x + y) * 2)")
print("Comparing: PyTorch (3 kernels) vs Custom CUDA (1 fused kernel)")

# Load CUDA extension
print("\n[SETUP] Loading CUDA extension...")
sys.stdout.flush()

try:
    from ops.cuda import add_mul_exp_cuda
    from ops.torch import add_mul_exp_pytorch

    print("✓ CUDA extension loaded")
except ImportError as e:
    print(f"✗ Failed to import ops: {e}")
    print("\nPlease install the package first:")
    print("  pip install --no-build-isolation -e .")
    sys.exit(1)


def main():
    """Run comprehensive benchmark."""
    # Configuration
    size = 10_000_000  # 10 million elements

    print(f"\n[SETUP] Test Configuration:")
    print(f"  Array size:     {size:,} elements")
    print(f"  Data type:      float32")
    print(f"  Memory/tensor:  {(size * 4) / (1024**2):.2f} MB")
    print(f"  Total memory:   {(size * 4 * 3) / (1024**2):.2f} MB (x, y, output)")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n✗ CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU:            {gpu_name}")

    # Allocate tensors
    print("\n[SETUP] Allocating tensors on GPU...")
    sys.stdout.flush()
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    print("✓ Tensors allocated")

    # Quick sanity check
    print("\n[SETUP] Testing CUDA kernel...")
    sys.stdout.flush()
    try:
        _ = add_mul_exp_cuda(x[:100], y[:100])
        print("✓ CUDA kernel working")
    except Exception as e:
        print(f"✗ CUDA kernel failed: {e}")
        sys.exit(1)

    # Benchmark both implementations
    baseline_result, optimized_result = compare_implementations(
        baseline_func=add_mul_exp_pytorch,
        optimized_func=add_mul_exp_cuda,
        args=(x, y),
        baseline_name="PyTorch",
        optimized_name="CUDA",
        warmup=10,
        iterations=100,
        verbose=True,
    )

    # Memory traffic analysis
    memory_stats = analyze_memory_traffic(
        tensor_size=size,
        baseline_reads=4,  # PyTorch reads: x, y, a, b
        baseline_writes=3,  # PyTorch writes: a, b, c
        optimized_reads=2,  # CUDA reads: x, y
        optimized_writes=1,  # CUDA writes: c
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

    # PyTorch profiler for kernel-level details
    profile_with_pytorch_profiler(
        [
            (add_mul_exp_pytorch, "PyTorch", (x, y)),
            (add_mul_exp_cuda, "CUDA", (x, y)),
        ],
        iterations=10,
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

    print(f"\n{'=' * 70}")
    print("Why is CUDA faster?")
    print("  PyTorch:     7 memory ops (read x,y,a,b; write a,b,c)")
    print("  CUDA Fused:  3 memory ops (read x,y; write c)")
    print("  Intermediate values stay in registers (400x faster!)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

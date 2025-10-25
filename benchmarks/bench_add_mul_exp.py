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
from utils.analysis import analyze_numerical_accuracy
from utils.performance import (
    compare_three_implementations,
    profile_with_pytorch_profiler,
)

# ============================================================================
# BENCHMARK CONFIGURATION - Modify this section for new kernels
# ============================================================================
OPERATION_NAME = "Add-Mul-Exp Fusion"
OPERATION_DESCRIPTION = "exp((x + y) * 2)"

# Display header
print("=" * 70)
print(f"CUDA KERNEL FUSION BENCHMARK: {OPERATION_NAME}")
print("=" * 70)
print(f"\nOperation: {OPERATION_DESCRIPTION}")
print("Comparing: PyTorch vs torch.compile vs Custom CUDA")

# Load CUDA extension
print("\n[SETUP] Loading CUDA extension...")
sys.stdout.flush()

try:
    from ops.cuda import add_mul_exp_cuda
    from ops.torch import add_mul_exp_pytorch

    # Three function references
    baseline_func = add_mul_exp_pytorch  # Raw PyTorch
    compiled_func = torch.compile(add_mul_exp_pytorch)  # torch.compile
    optimized_func = add_mul_exp_cuda  # Custom CUDA

    print("[OK] CUDA extension loaded")
except ImportError as e:
    print(f"[FAIL] Failed to import ops: {e}")
    print("\nPlease install the package first:")
    print("  pip install --no-build-isolation -e .")
    sys.exit(1)


def run_benchmark_for_size(size, size_name):
    """Run benchmark for a specific array size."""
    print(f"\n{'=' * 70}")
    print(f"TESTING {size_name} SCENARIO: {size:,} elements")
    print(f"{'=' * 70}")

    warmup_iterations = 10
    benchmark_iterations = 100
    profiler_iterations = 10

    # ========================================================================
    # SETUP
    # ========================================================================
    print(f"\n[SETUP] Test Configuration:")
    print(f"  Array size:     {size:,} elements")
    print(f"  Data type:      float32")
    print(f"  Memory/tensor:  {(size * 4) / (1024**2):.2f} MB")
    print(f"  Total memory:   {(size * 4 * 3) / (1024**2):.2f} MB (x, y, output)")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n[FAIL] CUDA not available!")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU:            {gpu_name}")

    # Prepare benchmark arguments
    print("\n[SETUP] Allocating tensors on GPU...")
    sys.stdout.flush()
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.randn(size, device="cuda", dtype=torch.float32)
    benchmark_args = (x, y)
    print("[OK] Tensors allocated")

    # Quick sanity check
    print("\n[SETUP] Testing CUDA kernel...")
    sys.stdout.flush()
    try:
        _ = optimized_func(x[:100], y[:100])
        print("[OK] CUDA kernel working")
    except Exception as e:
        print(f"[FAIL] CUDA kernel failed: {e}")
        sys.exit(1)

    # Warm up torch.compile for this specific size
    print("\n[SETUP] Warming up torch.compile for this size...")
    sys.stdout.flush()
    for _ in range(20):
        _ = compiled_func(x, y)
    torch.cuda.synchronize()
    print("[OK] torch.compile warmed up")

    # ========================================================================
    # BENCHMARKING (Performance Analysis)
    # ========================================================================
    pytorch_result, compiled_result, cuda_result = compare_three_implementations(
        baseline_func=baseline_func,
        compiled_func=compiled_func,
        optimized_func=optimized_func,
        args=benchmark_args,
        baseline_name="PyTorch",
        compiled_name="torch.compile",
        optimized_name="Custom CUDA",
        warmup=warmup_iterations,
        iterations=benchmark_iterations,
        verbose=True,
    )

    # Save timing metrics for summary (mean and std)
    pytorch_time = pytorch_result.mean_time_ms
    pytorch_std = pytorch_result.std_time_ms
    compiled_time = compiled_result.mean_time_ms
    compiled_std = compiled_result.std_time_ms
    cuda_time = cuda_result.mean_time_ms
    cuda_std = cuda_result.std_time_ms

    # ========================================================================
    # NUMERICAL ACCURACY ANALYSIS (Correctness)
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("NUMERICAL ACCURACY")
    print(f"{'=' * 70}")

    # Compare all three implementations (only checks correctness, not performance)
    accuracy_pytorch_vs_compiled = analyze_numerical_accuracy(
        baseline=pytorch_result.result,
        optimized=compiled_result.result,
        verbose=False,
    )
    accuracy_pytorch_vs_cuda = analyze_numerical_accuracy(
        baseline=pytorch_result.result,
        optimized=cuda_result.result,
        verbose=False,
    )
    accuracy_compiled_vs_cuda = analyze_numerical_accuracy(
        baseline=compiled_result.result,
        optimized=cuda_result.result,
        verbose=False,
    )

    print(f"\nPyTorch vs torch.compile:")
    print(f"  Max abs error: {accuracy_pytorch_vs_compiled.max_abs_diff:.6e}")
    print(f"  Max rel error: {accuracy_pytorch_vs_compiled.max_rel_diff:.6e}")
    print(
        f"  Identical:     {'YES' if accuracy_pytorch_vs_compiled.passes_1e5 else 'NO'}"
    )

    print(f"\nPyTorch vs Custom CUDA:")
    print(f"  Max abs error: {accuracy_pytorch_vs_cuda.max_abs_diff:.6e}")
    print(f"  Max rel error: {accuracy_pytorch_vs_cuda.max_rel_diff:.6e}")
    print(f"  Identical:     {'YES' if accuracy_pytorch_vs_cuda.passes_1e5 else 'NO'}")

    print(f"\ntorch.compile vs Custom CUDA:")
    print(f"  Max abs error: {accuracy_compiled_vs_cuda.max_abs_diff:.6e}")
    print(f"  Max rel error: {accuracy_compiled_vs_cuda.max_rel_diff:.6e}")
    print(f"  Identical:     {'YES' if accuracy_compiled_vs_cuda.passes_1e5 else 'NO'}")

    # Free result tensors to save memory for profiling
    del pytorch_result, compiled_result, cuda_result
    torch.cuda.empty_cache()

    # ========================================================================
    # PYTORCH PROFILER (Kernel-level timing)
    # ========================================================================
    profile_with_pytorch_profiler(
        [
            (baseline_func, "PyTorch", benchmark_args),
            (compiled_func, "torch.compile", benchmark_args),
            (optimized_func, "Custom CUDA", benchmark_args),
        ],
        iterations=profiler_iterations,
    )

    # Return timing results for summary table
    return {
        "pytorch_time": pytorch_time,
        "pytorch_std": pytorch_std,
        "compiled_time": compiled_time,
        "compiled_std": compiled_std,
        "cuda_time": cuda_time,
        "cuda_std": cuda_std,
        "passes": accuracy_pytorch_vs_cuda.passes_1e5
        and accuracy_compiled_vs_cuda.passes_1e5,
    }


def main():
    """Run comprehensive benchmark for multiple memory scenarios."""
    # Test multiple scenarios across L2 cache and VRAM
    # Note: For add_mul_exp, data size = 3x elements (2 inputs + 1 output)
    scenarios = [
        (100_000, "L2 Cache"),
        (300_000, "L2 Cache"),
        (1_000_000, "VRAM"),
        (5_000_000, "VRAM"),
        (20_000_000, "VRAM"),
    ]

    results = []
    for idx, (size, memory_location) in enumerate(scenarios):
        result = run_benchmark_for_size(size, memory_location)
        result["size"] = size
        result["memory_location"] = memory_location
        # Calculate data size: 3 tensors (x, y, output) × float32 (4 bytes)
        result["data_size_mb"] = (size * 3 * 4) / (1024**2)
        results.append(result)

        # Clean up memory between scenarios
        if idx < len(scenarios) - 1:
            print(f"\n[CLEANUP] Freeing GPU memory...")
            torch.cuda.empty_cache()
            import gc

            gc.collect()
            print("[OK] Memory cleaned")

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print(f"\n{'=' * 110}")
    print("PERFORMANCE SUMMARY")
    print(f"{'=' * 110}")

    print(f"\nRAW EXECUTION TIME")
    print(
        f"{'Memory Location':<18} {'Data Size':>12} {'# Elements':>15} {'PyTorch':>18} {'torch.compile':>18} {'Custom CUDA':>18}"
    )
    print(f"{'':18} {'(MB)':>12} {'':>15} {'(ms)':>18} {'(ms)':>18} {'(ms)':>18}")
    print("-" * 110)

    for r in results:
        elements_str = f"{r['size']:,}"
        data_size_str = f"{r['data_size_mb']:.1f}"
        pytorch_str = f"{r['pytorch_time']:>7.3f} ± {r['pytorch_std']:<5.3f}"
        compiled_str = f"{r['compiled_time']:>7.3f} ± {r['compiled_std']:<5.3f}"
        cuda_str = f"{r['cuda_time']:>7.3f} ± {r['cuda_std']:<5.3f}"
        print(
            f"{r['memory_location']:<18} {data_size_str:>12} {elements_str:>15} {pytorch_str:>18} {compiled_str:>18} {cuda_str:>18}"
        )

    print(f"\nRELATIVE SPEEDUP")
    print(
        f"{'Memory Location':<18} {'Data Size':>12} {'compile/PyTorch':>22} {'CUDA/PyTorch':>22} {'CUDA/compile':>22}"
    )
    print(f"{'':18} {'(MB)':>12} {'(speedup)':>22} {'(speedup)':>22} {'(speedup)':>22}")
    print("-" * 110)

    for r in results:
        import math

        data_size_str = f"{r['data_size_mb']:.1f}"

        # Calculate speedups
        compiled_speedup = r["pytorch_time"] / r["compiled_time"]
        cuda_speedup = r["pytorch_time"] / r["cuda_time"]
        cuda_vs_compiled = r["compiled_time"] / r["cuda_time"]

        # Error propagation: for A/B, σ = (A/B) * sqrt((σ_A/A)² + (σ_B/B)²)
        compiled_speedup_err = compiled_speedup * math.sqrt(
            (r["pytorch_std"] / r["pytorch_time"]) ** 2
            + (r["compiled_std"] / r["compiled_time"]) ** 2
        )
        cuda_speedup_err = cuda_speedup * math.sqrt(
            (r["pytorch_std"] / r["pytorch_time"]) ** 2
            + (r["cuda_std"] / r["cuda_time"]) ** 2
        )
        cuda_vs_compiled_err = cuda_vs_compiled * math.sqrt(
            (r["compiled_std"] / r["compiled_time"]) ** 2
            + (r["cuda_std"] / r["cuda_time"]) ** 2
        )

        compiled_str = f"{compiled_speedup:>6.2f} ± {compiled_speedup_err:<5.2f}x"
        cuda_str = f"{cuda_speedup:>6.2f} ± {cuda_speedup_err:<5.2f}x"
        cuda_comp_str = f"{cuda_vs_compiled:>6.2f} ± {cuda_vs_compiled_err:<5.2f}x"

        print(
            f"{r['memory_location']:<18} {data_size_str:>12} {compiled_str:>22} {cuda_str:>22} {cuda_comp_str:>22}"
        )

    # Check if all passed
    all_passed = all(r["passes"] for r in results)
    print(f"\n{'=' * 110}")
    print(
        f"Numerical Accuracy: {'PASS - All implementations match' if all_passed else 'FAIL - Results differ'}"
    )
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()

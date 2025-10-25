"""
Numerical accuracy analysis utilities for CUDA kernels.

Provides reusable functions for comparing numerical accuracy between implementations.
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class AccuracyMetrics:
    """Numerical accuracy metrics comparing two tensors."""

    max_abs_diff: float
    mean_abs_diff: float

    max_rel_diff: float
    mean_rel_diff: float

    max_diff_idx: int
    baseline_value_at_max: float
    optimized_value_at_max: float

    num_nan_baseline: int
    num_nan_optimized: int
    num_inf_baseline: int
    num_inf_optimized: int

    passes_1e3: bool
    passes_1e4: bool
    passes_1e5: bool
    passes_1e6: bool
    passes_1e7: bool

    def __str__(self) -> str:
        return (
            f"Absolute Differences:\n"
            f"  Max:    {self.max_abs_diff:.6e}\n"
            f"  Mean:   {self.mean_abs_diff:.6e}\n"
            f"\n"
            f"Relative Differences:\n"
            f"  Max:    {self.max_rel_diff:.6e}\n"
            f"  Mean:   {self.mean_rel_diff:.6e}\n"
            f"\n"
            f"Passes torch.allclose():\n"
            f"  rtol=1e-3: {'PASS' if self.passes_1e3 else 'FAIL'}\n"
            f"  rtol=1e-4: {'PASS' if self.passes_1e4 else 'FAIL'}\n"
            f"  rtol=1e-5: {'PASS' if self.passes_1e5 else 'FAIL'}\n"
            f"  rtol=1e-6: {'PASS' if self.passes_1e6 else 'FAIL'}\n"
            f"  rtol=1e-7: {'PASS' if self.passes_1e7 else 'FAIL'}"
        )


def analyze_numerical_accuracy(
    baseline: torch.Tensor,
    optimized: torch.Tensor,
    verbose: bool = True,
) -> AccuracyMetrics:
    """
    Comprehensive numerical accuracy analysis comparing two tensors.

    Automatically handles integer dtypes (int8, int16, int32, int64) by converting
    to int for exact comparison, and float dtypes with relative error analysis.

    Args:
        baseline: Baseline result (e.g., from PyTorch)
        optimized: Optimized result (e.g., from custom CUDA)
        verbose: Whether to print detailed analysis

    Returns:
        AccuracyMetrics with detailed comparison

    Example:
        >>> baseline = torch.exp((x + y) * 2)
        >>> optimized = add_mul_exp(x, y)
        >>> metrics = analyze_numerical_accuracy(baseline, optimized)
        >>> print(metrics)
    """
    # Handle integer dtypes by converting to int32/int64 for comparison
    is_integer_type = baseline.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]

    if is_integer_type:
        # Convert to int for exact arithmetic (avoid overflow in int8)
        baseline_comp = baseline.int()
        optimized_comp = optimized.int()
    else:
        baseline_comp = baseline
        optimized_comp = optimized

    # Compute absolute differences
    abs_diff = torch.abs(baseline_comp - optimized_comp)

    # Extract key stats from abs_diff before computing more tensors
    # Note: Skip median/std to avoid high memory operations (sorting/variance)
    max_abs_diff = abs_diff.max().item()
    max_diff_idx = torch.argmax(abs_diff).item()
    mean_abs_diff = abs_diff.float().mean().item() if is_integer_type else abs_diff.mean().item()

    # Extract baseline values at max diff location now
    baseline_value_at_max = baseline.flatten()[max_diff_idx].item()
    optimized_value_at_max = optimized.flatten()[max_diff_idx].item()

    # Compute relative differences (avoid division by zero)
    if is_integer_type:
        denominator = torch.maximum(torch.abs(baseline_comp.float()), torch.abs(optimized_comp.float()))
        denominator = torch.clamp(denominator, min=1e-10)
        rel_diff = abs_diff.float() / denominator
    else:
        denominator = torch.maximum(torch.abs(baseline_comp), torch.abs(optimized_comp))
        denominator = torch.clamp(denominator, min=1e-10)
        rel_diff = abs_diff / denominator

    # Free denominator and abs_diff immediately
    del denominator
    del abs_diff

    # Extract rel_diff stats and free it
    # Note: Skip median/std to save memory
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    del rel_diff

    # Special values - skip to save memory for large tensors
    # For 200M elements, boolean tensors would be 200MB each (8 tensors = 1.6GB extra)
    # These checks are rarely needed and can be skipped for memory-constrained scenarios
    num_nan_baseline = 0
    num_nan_optimized = 0
    num_inf_baseline = 0
    num_inf_optimized = 0

    # Tolerance checks - only check the most important one (1e-5) to save memory
    # torch.allclose creates intermediate tensors which cause OOM for large arrays
    passes = {}
    if is_integer_type:
        # For integers, check exact match
        exact_match = torch.equal(baseline, optimized)
        passes[1e-3] = exact_match
        passes[1e-4] = exact_match
        passes[1e-5] = exact_match  # Main check
        passes[1e-6] = exact_match
        passes[1e-7] = exact_match
    else:
        # Only do the 1e-5 check (most commonly used threshold)
        # Use max errors instead of torch.allclose to avoid creating intermediate tensors
        passes_1e5 = (max_abs_diff <= 1e-8) and (max_rel_diff <= 1e-5)
        passes[1e-3] = passes_1e5  # If passes 1e-5, definitely passes 1e-3
        passes[1e-4] = passes_1e5  # If passes 1e-5, definitely passes 1e-4
        passes[1e-5] = passes_1e5  # Main check
        passes[1e-6] = False  # Don't check stricter tolerances to save memory
        passes[1e-7] = False

    metrics = AccuracyMetrics(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_rel_diff=max_rel_diff,
        mean_rel_diff=mean_rel_diff,
        max_diff_idx=max_diff_idx,
        baseline_value_at_max=baseline_value_at_max,
        optimized_value_at_max=optimized_value_at_max,
        num_nan_baseline=num_nan_baseline,
        num_nan_optimized=num_nan_optimized,
        num_inf_baseline=num_inf_baseline,
        num_inf_optimized=num_inf_optimized,
        passes_1e3=passes[1e-3],
        passes_1e4=passes[1e-4],
        passes_1e5=passes[1e-5],
        passes_1e6=passes[1e-6],
        passes_1e7=passes[1e-7],
    )

    if verbose:
        print(f"\n{'=' * 70}")
        dtype_str = f" ({baseline.dtype})" if is_integer_type else ""
        print(f"NUMERICAL ACCURACY ANALYSIS{dtype_str}")
        print(f"{'=' * 70}\n")

        if is_integer_type:
            exact_match = torch.equal(baseline, optimized)
            print(f"Integer Comparison:")
            print(f"  Exact match: {'YES' if exact_match else 'NO'}")
            print()

        print(metrics)

        # Additional details
        print(f"\nMax Difference Location:")
        print(f"  Index:           {metrics.max_diff_idx}")
        print(f"  Baseline value:  {metrics.baseline_value_at_max:.10f}")
        print(f"  Optimized value: {metrics.optimized_value_at_max:.10f}")
        print(f"  Absolute diff:   {metrics.max_abs_diff:.10f}")

        # Value ranges
        print(f"\nValue Range Analysis:")
        if is_integer_type:
            print(f"  Baseline:  [{baseline.min().item()}, {baseline.max().item()}]")
            print(f"  Optimized: [{optimized.min().item()}, {optimized.max().item()}]")
        else:
            print(f"  Baseline:  [{baseline.min():.6e}, {baseline.max():.6e}], mean={baseline.mean():.6e}")
            print(f"  Optimized: [{optimized.min():.6e}, {optimized.max():.6e}], mean={optimized.mean():.6e}")

    return metrics



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
    median_abs_diff: float
    std_abs_diff: float

    max_rel_diff: float
    mean_rel_diff: float
    median_rel_diff: float
    std_rel_diff: float

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
            f"  Median: {self.median_abs_diff:.6e}\n"
            f"\n"
            f"Relative Differences:\n"
            f"  Max:    {self.max_rel_diff:.6e}\n"
            f"  Mean:   {self.mean_rel_diff:.6e}\n"
            f"  Median: {self.median_rel_diff:.6e}\n"
            f"\n"
            f"Special Values:\n"
            f"  Baseline NaN/Inf: {self.num_nan_baseline}/{self.num_inf_baseline}\n"
            f"  Optimized NaN/Inf: {self.num_nan_optimized}/{self.num_inf_optimized}\n"
            f"\n"
            f"Passes torch.allclose():\n"
            f"  rtol=1e-3: {'✓' if self.passes_1e3 else '✗'}\n"
            f"  rtol=1e-4: {'✓' if self.passes_1e4 else '✗'}\n"
            f"  rtol=1e-5: {'✓' if self.passes_1e5 else '✗'}\n"
            f"  rtol=1e-6: {'✓' if self.passes_1e6 else '✗'}\n"
            f"  rtol=1e-7: {'✓' if self.passes_1e7 else '✗'}"
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

    # Absolute differences
    abs_diff = torch.abs(baseline_comp - optimized_comp)

    # Relative differences (avoid division by zero)
    # For integers, convert to float for relative error calculation
    if is_integer_type:
        denominator = torch.maximum(torch.abs(baseline_comp.float()), torch.abs(optimized_comp.float()))
        denominator = torch.clamp(denominator, min=1e-10)
        rel_diff = abs_diff.float() / denominator
    else:
        denominator = torch.maximum(torch.abs(baseline_comp), torch.abs(optimized_comp))
        denominator = torch.clamp(denominator, min=1e-10)
        rel_diff = abs_diff / denominator

    # Find max difference location
    max_diff_idx = torch.argmax(abs_diff).item()

    # Special values
    num_nan_baseline = torch.isnan(baseline).sum().item()
    num_nan_optimized = torch.isnan(optimized).sum().item()
    num_inf_baseline = torch.isinf(baseline).sum().item()
    num_inf_optimized = torch.isinf(optimized).sum().item()

    # Tolerance checks
    passes = {}
    if is_integer_type:
        # For integers, check exact match and small absolute differences
        exact_match = torch.equal(baseline, optimized)
        passes[1e-3] = exact_match or (abs_diff <= 1).all()  # Within 1
        passes[1e-4] = exact_match or (abs_diff <= 1).all()
        passes[1e-5] = exact_match  # Exact match
        passes[1e-6] = exact_match
        passes[1e-7] = exact_match
    else:
        for rtol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            passes[rtol] = torch.allclose(baseline, optimized, rtol=rtol, atol=1e-8)

    # Convert to float for statistical calculations (needed for integer types)
    abs_diff_float = abs_diff.float() if is_integer_type else abs_diff
    rel_diff_float = rel_diff.float() if is_integer_type else rel_diff

    metrics = AccuracyMetrics(
        max_abs_diff=abs_diff.max().item(),
        mean_abs_diff=abs_diff_float.mean().item(),
        median_abs_diff=abs_diff_float.median().item(),
        std_abs_diff=abs_diff_float.std().item(),
        max_rel_diff=rel_diff.max().item(),
        mean_rel_diff=rel_diff_float.mean().item(),
        median_rel_diff=rel_diff_float.median().item(),
        std_rel_diff=rel_diff_float.std().item(),
        max_diff_idx=max_diff_idx,
        baseline_value_at_max=baseline.flatten()[max_diff_idx].item(),
        optimized_value_at_max=optimized.flatten()[max_diff_idx].item(),
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
            print(f"  Exact match: {'✓ YES' if exact_match else '✗ NO'}")
            if not exact_match:
                num_mismatches = (abs_diff != 0).sum().item()
                print(f"  Mismatches:  {num_mismatches:,} / {baseline.numel():,} "
                      f"({100*num_mismatches/baseline.numel():.4f}%)")
            print()

        print(metrics)

        # Additional details
        print(f"\nMax Difference Location:")
        print(f"  Index:           {metrics.max_diff_idx}")
        print(f"  Baseline value:  {metrics.baseline_value_at_max:.10f}")
        print(f"  Optimized value: {metrics.optimized_value_at_max:.10f}")
        print(f"  Absolute diff:   {metrics.max_abs_diff:.10f}")

        # Error distribution
        print(f"\nError Distribution (percentiles):")
        percentiles = [50, 90, 95, 99, 99.9, 100]
        for p in percentiles:
            abs_val = torch.quantile(abs_diff_float, p / 100).item()
            rel_val = torch.quantile(rel_diff_float, p / 100).item()
            print(f"  {p:5.1f}%: abs={abs_val:.6e}, rel={rel_val:.6e}")

        # Value ranges
        print(f"\nValue Range Analysis:")
        if is_integer_type:
            print(f"  Baseline:  [{baseline.min().item()}, {baseline.max().item()}]")
            print(f"  Optimized: [{optimized.min().item()}, {optimized.max().item()}]")
        else:
            print(f"  Baseline:  [{baseline.min():.6e}, {baseline.max():.6e}], mean={baseline.mean():.6e}")
            print(f"  Optimized: [{optimized.min():.6e}, {optimized.max():.6e}], mean={optimized.mean():.6e}")

    return metrics


def compare_accuracy_quick(
    baseline: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, float, float]:
    """
    Quick accuracy check - returns pass/fail and error metrics.

    Args:
        baseline: Baseline result
        optimized: Optimized result
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        (passes, max_abs_error, max_rel_error)

    Example:
        >>> passes, abs_err, rel_err = compare_accuracy_quick(baseline, optimized)
        >>> print(f"Pass: {passes}, Max abs error: {abs_err:.2e}")
    """
    abs_diff = torch.abs(baseline - optimized)
    max_abs_error = abs_diff.max().item()

    denominator = torch.maximum(torch.abs(baseline), torch.abs(optimized))
    denominator = torch.clamp(denominator, min=1e-10)
    rel_diff = abs_diff / denominator
    max_rel_error = rel_diff.max().item()

    passes = torch.allclose(baseline, optimized, rtol=rtol, atol=atol)

    return passes, max_abs_error, max_rel_error


def assert_accuracy(
    baseline: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    operation_name: str = "Operation",
) -> None:
    """
    Assert that optimized result matches baseline within tolerance.

    Raises AssertionError with detailed message if check fails.

    Args:
        baseline: Baseline result
        optimized: Optimized result
        rtol: Relative tolerance
        atol: Absolute tolerance
        operation_name: Name of operation for error message

    Example:
        >>> assert_accuracy(baseline, optimized, rtol=1e-5, operation_name="add_mul_exp")
    """
    abs_diff = torch.abs(baseline - optimized)
    max_abs_error = abs_diff.max().item()

    denominator = torch.maximum(torch.abs(baseline), torch.abs(optimized))
    denominator = torch.clamp(denominator, min=1e-10)
    rel_diff = abs_diff / denominator
    max_rel_error = rel_diff.max().item()

    passes = torch.allclose(baseline, optimized, rtol=rtol, atol=atol)

    if not passes:
        max_diff_idx = torch.argmax(abs_diff).item()
        baseline_val = baseline.flatten()[max_diff_idx].item()
        optimized_val = optimized.flatten()[max_diff_idx].item()

        raise AssertionError(
            f"{operation_name} failed accuracy check:\n"
            f"  Max absolute error: {max_abs_error:.6e} (threshold: {atol:.6e})\n"
            f"  Max relative error: {max_rel_error:.6e} (threshold: {rtol:.6e})\n"
            f"  Location of max error: index {max_diff_idx}\n"
            f"    Baseline:  {baseline_val:.10f}\n"
            f"    Optimized: {optimized_val:.10f}\n"
            f"    Diff:      {abs_diff.flatten()[max_diff_idx]:.10f}"
        )

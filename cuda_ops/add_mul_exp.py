"""
Python interface to CUDA kernels.

This file:
1. Compiles the CUDA code (JIT = Just-In-Time compilation)
2. Provides a Python function that calls the CUDA kernel
"""

import os

import torch
from torch.utils.cpp_extension import load

# Get the directory where this file lives
current_dir = os.path.dirname(os.path.abspath(__file__))

# JIT compile the CUDA extension
# This happens the first time you import this module
# Subsequent imports use the cached compiled version
_C = load(
    name="add_mul_exp_cuda",  # Name for the compiled module
    sources=[
        os.path.join(current_dir, "csrc", "add_mul_exp.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",  # Optimization level 3 (maximum)
        "--use_fast_math",  # Use faster (slightly less accurate) math
    ],
    verbose=True,  # Print compilation output (useful for debugging)
)


def add_mul_exp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Fused operation: c = exp((x + y) * 2)

    This is a Python function that calls our CUDA kernel.

    Args:
        x: Input tensor (must be on CUDA, float32)
        y: Input tensor (must be on CUDA, float32, same shape as x)

    Returns:
        Output tensor: exp((x + y) * 2)

    Raises:
        ValueError: If inputs are not CUDA tensors
        RuntimeError: If CUDA kernel fails
    """
    # Input validation
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not y.is_cuda:
        raise ValueError("y must be a CUDA tensor")
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x{x.shape} vs y{y.shape}")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError("Only float32 tensors supported")

    # Call the C++ wrapper, which launches the CUDA kernel
    # _C.fused_add_mul_exp is defined in fusion_kernel.cu via PYBIND11_MODULE
    return _C.add_mul_exp(x, y)

"""
Python interface to CUDA kernels.

This file:
1. Compiles the CUDA code (JIT = Just-In-Time compilation)
2. Provides Python functions that call the CUDA kernel
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
    name="quantize_int8_cuda",  # Name for the compiled module
    sources=[
        os.path.join(current_dir, "csrc", "quantize_int8.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",  # Optimization level 3 (maximum)
        "--use_fast_math",  # Use faster (slightly less accurate) math
    ],
    verbose=True,  # Print compilation output (useful for debugging)
    with_cuda=True,  # Explicitly enable CUDA
)


def quantize_int8_cuda(
    x: torch.Tensor,
    scale: float,
    zero_point: float,
) -> torch.Tensor:
    """
    Unfused PyTorch implementation of int8 quantization.

    This is the baseline reference implementation used for:
    - Correctness verification in tests
    - Performance comparison in benchmarks

    Args:
        x: Input tensor (any shape, must be on CUDA)
        scale: Input float that already includes 128.0 multiplier for int8
            ex. scale = original_scale * 128.0
        zero_point: Offset for asymmetric quantization

    Returns:
        Tensor with same shape as x quantized to int8
    """
    # Input validation
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.float32:
        raise ValueError("Only float32 tensors supported")

    # Call the C++ wrapper, which launches the CUDA kernel
    return _C.quantize_int8(x, scale, zero_point)


def quantize_int8(
    x: torch.Tensor,
    scale: float,
    zero_point: float,
) -> torch.Tensor:
    """
    Unfused PyTorch implementation of int8 quantization.

    This is the baseline reference implementation used for:
    - Correctness verification in tests
    - Performance comparison in benchmarks

    Args:
        x: Input tensor (any shape, CUDA or CPU)
        scale: Input float that already includes 128.0 multiplier for int8
            ex. scale = original_scale * 128.0
        zero_point: Offset for asymmetric quantization

    Returns:
        Tensor with same shape as x quantized to int8

    Example:
        >>> x = torch.randn(1000, device='cuda')
        >>> scale = 4.2
        >>> zero_point = 1.2
        >>> result = quantize_int8_pytorch(x, scale, zero_point)
        >>> # Equivalent to:
        >>> # a = x / scale
        >>> # b = a + zero_point
        >>> # c = b.round()
        >>> # d = c.clamp(-128, 127)
        >>> # result = d.to(torch.int8)
    """
    return quantize_int8_cuda(x, scale, zero_point)

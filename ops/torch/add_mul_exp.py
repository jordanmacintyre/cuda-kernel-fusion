"""
PyTorch baseline implementation for add_mul_exp operation.

Operation: exp((x + y) * 2)

This unfused version launches 3 separate CUDA kernels:
1. Addition:       a = x + y
2. Multiplication: b = a * 2
3. Exponential:    c = exp(b)

Total memory operations: 7
- Reads:  x, y, a, b (4 reads)
- Writes: a, b, c    (3 writes)
"""

import torch


def add_mul_exp_pytorch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Unfused PyTorch implementation of exp((x + y) * 2).

    This is the baseline reference implementation used for:
    - Correctness verification in tests
    - Performance comparison in benchmarks

    Args:
        x: Input tensor (any shape, CUDA or CPU)
        y: Input tensor (same shape as x)

    Returns:
        Tensor with same shape as inputs containing exp((x + y) * 2)

    Example:
        >>> x = torch.randn(1000, device='cuda')
        >>> y = torch.randn(1000, device='cuda')
        >>> result = add_mul_exp_pytorch(x, y)
        >>> # Equivalent to:
        >>> # a = x + y
        >>> # b = a * 2
        >>> # result = torch.exp(b)
    """
    # Kernel 1: Element-wise addition
    a = x + y

    # Kernel 2: Element-wise multiplication by scalar
    b = a * 2

    # Kernel 3: Element-wise exponential
    c = torch.exp(b)

    return c

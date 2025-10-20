"""
PyTorch baseline implementation for quantize_int8 operation (asymmetric quantization).
FP32 -> INT8 with scaling, zero_point offset, rounding, clamping, dtype conversion

Operation: ((x / scale) + zero_point).round().clamp(-128, 127).to(torch.int8)

This unfused version launches 5 separate CUDA kernels:
1. Division:        a = x / scale
2. Addition:        b = a + zero_point
3. Rounding:        c = b.round()
4. Clamp:           d = c.clamp(-128, 127)
5. Type Conversion: result = d.to(torch.int8)

Total memory operations: 7
- Reads:  x, a, b, c, d      (5 reads)
- Writes: a, b, c, d, result (5 writes)
"""

import torch


def quantize_int8_pytorch(
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
    # Kernel 1: Element-wise division by scalar
    a = x / scale

    # Kernel 2: Element-wise addition with scalar
    b = a + zero_point

    # Kernel 3: Rounding
    c = b.round()

    # Kernel 4: Clamp values
    d = c.clamp(-128, 127)

    # Kernel 5: Covert data type
    result = d.to(torch.int8)

    return result

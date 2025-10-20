"""
PyTorch baseline implementations for CUDA operations.

These are unfused reference implementations using PyTorch operations.
Used for correctness testing and performance comparison.
"""

from .add_mul_exp import add_mul_exp_pytorch
from .quantize_int8 import quantize_int8_pytorch

__all__ = [
    "add_mul_exp_pytorch",
    "quantize_int8_pytorch",
]

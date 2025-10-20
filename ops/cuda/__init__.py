"""
CUDA operations package.
Exposes custom CUDA kernels.

Naming convention:
- operation_name_cuda(): CUDA implementation
- operation_name_pytorch(): PyTorch baseline (in top-level baselines/ package)
- operation_name(): Main API (dispatches to best available)
"""

from .add_mul_exp import add_mul_exp, add_mul_exp_cuda
from .quantize_int8 import quantize_int8, quantize_int8_cuda

__all__ = [
    "add_mul_exp",  # Main API
    "add_mul_exp_cuda",  # CUDA implementation
    "quantize_int8",  # Main API
    "quantize_int8_cuda",  # CUDA implementation
]

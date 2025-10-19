"""
Operations package with multiple backend implementations.

Subpackages:
- ops.cuda: CUDA kernel implementations
- ops.torch: PyTorch reference implementations
- ops.triton: (future) Triton kernel implementations

Naming convention:
- operation_cuda(): CUDA implementation
- operation_pytorch(): PyTorch implementation
- operation_triton(): Triton implementation
- operation(): Main API (dispatches to best available)
"""

# Re-export main APIs from backends
from .cuda import add_mul_exp, add_mul_exp_cuda

__all__ = [
    "add_mul_exp",
    "add_mul_exp_cuda",
]

/*
 * Header file for CUDA fusion kernels.
 * 
 * This declares the interface that other files can use.
 * In this simple example, only fusion_kernel.cu uses it,
 * but in larger projects, multiple .cu files might share declarations.
 */

#pragma once  // Prevent multiple inclusion

#include <torch/extension.h>  // PyTorch's C++ API

/*
 * Declare the C++ function that Python will call.
 * 
 * This function:
 * 1. Takes two PyTorch tensors as input
 * 2. Launches our CUDA kernel
 * 3. Returns the result tensor
 */
torch::Tensor add_mul_exp_cuda(
    torch::Tensor x,
    torch::Tensor y
);
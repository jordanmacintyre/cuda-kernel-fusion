/*
 * Header file for CUDA fusion kernels.
 * 
 * This declares the interface that other files can use.
 */

#pragma once  // Prevent multiple inclusion

#include <torch/extension.h>  // PyTorch's C++ API

/*
 * Declare the C++ function that Python will call.
 * 
 * This function:
 * 1. Takes one PyTorch tensor and 2 floats as input
 * 2. Launches CUDA kernel
 * 3. Returns the result tensor
 */
torch::Tensor quantize_int8_cuda(
    torch::Tensor x,
    float scale,
    float zero_point
);
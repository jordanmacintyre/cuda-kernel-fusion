/*
 * CUDA Kernel Implementation
 * 
 * This file contains:
 * 1. The actual CUDA kernel (__global__ function that runs on GPU)
 * 2. C++ wrapper that launches the kernel
 * 3. Python binding to expose it to Python
 */

#include "quantize_int8.h"
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * ============================================================================
 * THE CUDA KERNEL - This runs on the GPU!
 * ============================================================================
 * 
 * __global__ = This function runs on GPU, called from CPU
 * 
 */
__global__ void quantize_int8_kernel(
    const float* __restrict__ x, // Input array x (read-only)
    int8_t* __restrict__ output, // Output array (write)
    float scale,                 // Scaling factor (including 128.0 multiplier for int8 quantization)
    float zero_point,            // Offset for asymmetric quantization
    int size                     // Number of elements
) {
    /*
     * THREAD INDEXING - Calculate which element this thread processes
     *
     * blockIdx.x  = Which block this thread is in (e.g., 0, 1, 2, ...)
     * blockDim.x  = Number of threads per block (e.g., 256)
     * threadIdx.x = Thread position within its block (e.g., 0-255)
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * BOUNDS CHECK - Make sure we don't go past the array end
     */
    if (idx < size) {
        float a = x[idx] / scale;      // Divide (in register)
        float b = a + zero_point;      // Add (in register)
        int c = __float2int_rn(b);     // Round (in register)
        int d = max(-128, min(127, c));// Clamp (in register)
        output[idx] = (int8_t)d;       // Typecast and write to memory
    }
}

/*
 * ============================================================================
 * C++ WRAPPER - Launches the kernel from CPU
 * ============================================================================
 * 
 * This function:
 * 1. Runs on CPU
 * 2. Validates inputs
 * 3. Allocates output memory
 * 4. Configures kernel launch parameters
 * 5. "Launches" the kernel (tells GPU to start running it)
 * 6. Returns result
 */
torch::Tensor quantize_int8_cuda(
    torch::Tensor x,
    float scale,
    float zero_point
) {
    /*
     * INPUT VALIDATION - Catch errors before they cause crashes
     */
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    
    /*
     * ALLOCATE OUTPUT - Create tensor with same shape as input
     * Create int8 output tensor with same shape as input
     */
    auto output = torch::empty(x.sizes(), torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    
    /*
     * GET TOTAL NUMBER OF ELEMENTS
     * For a tensor of shape [1000, 500], numel() returns 500,000
     */
    int size = x.numel();
    
    /*
     * CONFIGURE KERNEL LAUNCH - How many threads to use?
     * 
     * Why 256? Common choices: 128, 256, 512, 1024
     * - Must be multiple of 32 (GPU "warp" size)
     * - 256 is a good balance for most problems
     * - Too few: GPU not fully utilized
     * - Too many: Not enough registers per thread
     */
    const int threads = 256;  // Threads per block
    const int blocks = (size + threads - 1) / threads;  // Ceiling division
    
    /*
     * LAUNCH THE KERNEL
     */
    quantize_int8_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),        // Get raw pointer to x's data
        output.data_ptr<int8_t>(),  // Get raw pointer to output's data
        scale,                      // Scaling factor (including 128.0 multiplier for int8 quantization)
        zero_point,                 // Offset for asymmetric quantization
        size                        // Number of elements
    );
    
    /*
     * ERROR CHECKING - Did the kernel launch successfully?
     * 
     * Note: This only checks launch errors, not execution errors!
     * For execution errors, you'd need cudaDeviceSynchronize()
     */
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "CUDA kernel failed: ", cudaGetErrorString(err)
    );
    
    /*
     * RETURN RESULT
     * The kernel execution happens asynchronously on GPU
     * PyTorch will automatically synchronize when needed
     */
    return output;
}

/*
 * ============================================================================
 * PYTHON BINDING - Make this callable from Python
 * ============================================================================
 * 
 * PYBIND11_MODULE creates a Python module from C++ code
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "quantize_int8",             // Python function name
        &quantize_int8_cuda,         // C++ function to call
        "Fused quantize+int8 (CUDA)" // Docstring
    );
}
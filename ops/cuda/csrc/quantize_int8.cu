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
     * Each thread processes 4 elements for optimal memory coalescing
     * At 50M elements, this kernel is MEMORY-BOUND, not compute-bound
     */
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    // Reciprocal scale for faster multiply instead of divide
    const float inv_scale = 1.0f / scale;

    // Inline quantization - clamp to int8 range [-128, 127]
    #define QUANTIZE(val) \
        (int8_t)max(-128, min(127, __float2int_rn(val * inv_scale + zero_point)))

    // Process 4 elements if they all fit in bounds
    if (idx + 3 < size) {
        // Vectorized load: 16 bytes (4 floats) in single transaction
        float4 data = reinterpret_cast<const float4*>(x)[idx / 4];

        // Process 4 elements
        char4 result;
        result.x = QUANTIZE(data.x);
        result.y = QUANTIZE(data.y);
        result.z = QUANTIZE(data.z);
        result.w = QUANTIZE(data.w);

        // Vectorized store: 4 bytes in single transaction
        reinterpret_cast<char4*>(output)[idx / 4] = result;
    }
    // Handle tail elements (when size is not multiple of 4)
    else if (idx < size) {
        for (int i = idx; i < size && i < idx + 4; i++) {
            output[i] = QUANTIZE(x[i]);
        }
    }

    #undef QUANTIZE
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
    
    int size = x.numel();

    /*
     * CONFIGURE KERNEL LAUNCH
     *
     * Each thread processes 4 elements (vectorized with float4/char4)
     * 256 threads/block = 8 warps, good occupancy
     */
    const int threads = 256;
    const int elements_per_thread = 4;
    const int total_threads_needed = (size + elements_per_thread - 1) / elements_per_thread;
    const int blocks = (total_threads_needed + threads - 1) / threads;
    
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
/*
 * CUDA Kernel Implementation
 * 
 * This file contains:
 * 1. The actual CUDA kernel (__global__ function that runs on GPU)
 * 2. C++ wrapper that launches the kernel
 * 3. Python binding to expose it to Python
 */

#include "add_mul_exp.h"
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * ============================================================================
 * THE CUDA KERNEL - This runs on the GPU!
 * ============================================================================
 * 
 * __global__ = This function runs on GPU, called from CPU
 * 
 * CUDA Programming Model:
 * - You launch thousands/millions of threads in parallel
 * - Each thread has a unique ID (threadIdx, blockIdx)
 * - Each thread processes a small piece of data
 * 
 * Example: For 1,000,000 elements
 * - Launch 3,907 blocks of 256 threads each = 1,000,192 threads
 * - Thread 0 processes element 0
 * - Thread 1 processes element 1
 * - ... and so on
 */
__global__ void add_mul_exp_kernel(
    const float* __restrict__ x,      // Input array x (read-only)
    const float* __restrict__ y,      // Input array y (read-only)
    float* __restrict__ output,       // Output array (write)
    int size                          // Number of elements
) {
    /*
     * THREAD INDEXING - Calculate which element this thread processes
     * 
     * blockIdx.x  = Which block this thread is in (e.g., 0, 1, 2, ...)
     * blockDim.x  = Number of threads per block (e.g., 256)
     * threadIdx.x = Thread position within its block (e.g., 0-255)
     * 
     * Example with 256 threads per block:
     * - Block 0, Thread 0:   idx = 0 * 256 + 0   = 0
     * - Block 0, Thread 1:   idx = 0 * 256 + 1   = 1
     * - Block 0, Thread 255: idx = 0 * 256 + 255 = 255
     * - Block 1, Thread 0:   idx = 1 * 256 + 0   = 256
     * - Block 1, Thread 1:   idx = 1 * 256 + 1   = 257
     * ... and so on
     */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    /*
     * BOUNDS CHECK - Make sure we don't go past the array end
     * 
     * Why? If size=1,000,000 and we launch 1,000,192 threads,
     * threads 1,000,000-1,000,192 would access invalid memory.
     */
    if (idx < size) {
        /*
         * THE ACTUAL COMPUTATION - All 3 operations in one place!
         * 
         * Key insight: a and b are stored in REGISTERS (super fast!)
         * - Registers are tiny storage locations on the GPU chip
         * - Access time: ~1 nanosecond
         * - Compare to global memory: ~400 nanoseconds
         * 
         * Memory traffic:
         * - Read x[idx] from global memory (slow)
         * - Read y[idx] from global memory (slow)
         * - Compute a = x + y (stored in register, FAST!)
         * - Compute b = a * 2.0f (stored in register, FAST!)
         * - Compute result = exp(b) (stored in register, FAST!)
         * - Write output[idx] to global memory (slow)
         * 
         * Total global memory accesses: 3 (2 reads + 1 write)
         * Compare to unfused PyTorch: 6 (3 reads + 3 writes)
         */
        float a = x[idx] + y[idx];      // Add (in register)
        float b = a * 2.0f;              // Multiply (in register)
        output[idx] = expf(b);           // Exp and write to memory
        
        /*
         * expf() = Fast single-precision exponential function
         * - Provided by CUDA math library
         * - Hardware-accelerated on GPU
         * - Use expf() for float, exp() for double
         */
    }
}

/*
 * About __restrict__:
 * 
 * This tells the compiler: "I promise these pointers don't overlap"
 * - x, y, and output point to different memory locations
 * - Compiler can optimize better knowing this
 * - Without it: compiler assumes pointers might alias (overlap)
 * - With it: compiler can reorder/optimize more aggressively
 * 
 * Example of what __restrict__ prevents:
 *   float* a = data;
 *   float* b = data;  // Same memory!
 *   a[0] = 1.0f;
 *   b[0] = 2.0f;      // Overwrites a[0]!
 * 
 * With __restrict__, you promise this won't happen.
 */

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
torch::Tensor add_mul_exp_cuda(
    torch::Tensor x,
    torch::Tensor y
) {
    /*
     * INPUT VALIDATION - Catch errors before they cause crashes
     */
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have same shape");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    
    /*
     * ALLOCATE OUTPUT - Create tensor with same shape as input
     * empty_like() doesn't initialize values (faster than zeros_like)
     */
    auto output = torch::empty_like(x);
    
    /*
     * GET TOTAL NUMBER OF ELEMENTS
     * For a tensor of shape [1000, 500], numel() returns 500,000
     */
    int size = x.numel();
    
    /*
     * CONFIGURE KERNEL LAUNCH - How many threads to use?
     * 
     * GPU organization:
     * - Threads are grouped into "blocks"
     * - Blocks are grouped into a "grid"
     * 
     * Example: 1,000,000 elements
     * - threads = 256 (threads per block)
     * - blocks = (1,000,000 + 255) / 256 = 3,907 blocks
     * - Total threads = 3,907 * 256 = 1,000,192 threads
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
     * LAUNCH THE KERNEL!
     * 
     * Syntax: kernel_name<<<blocks, threads>>>(args)
     *                      ^^^^^^^^^^^^^^^^^^^
     *                      This is CUDA-specific syntax
     * 
     * What "launching" means:
     * 1. CPU prepares the parameters (x, y, output, size)
     * 2. CPU sends these to GPU memory
     * 3. CPU tells GPU: "Run this kernel with these params"
     * 4. GPU schedules the blocks across its streaming multiprocessors
     * 5. GPU executes thousands of threads in parallel
     * 6. CPU continues executing (async!) or waits (sync)
     * 
     * Triple angle brackets <<<>>> are special CUDA syntax
     * The compiler translates this to:
     *   cudaConfigureCall(blocks, threads, ...);
     *   cudaSetupArgument(...);
     *   cudaLaunch(kernel);
     */
    add_mul_exp_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),      // Get raw pointer to x's data
        y.data_ptr<float>(),      // Get raw pointer to y's data
        output.data_ptr<float>(), // Get raw pointer to output's data
        size                      // Number of elements
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
 * 
 * After compilation, in Python you can do:
 *   import fusion_cuda
 *   result = add_mul_exp.add_mul_exp(x, y)
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "add_mul_exp",           // Python function name
        &add_mul_exp_cuda,       // C++ function to call
        "Fused add+mul+exp (CUDA)"     // Docstring
    );
}

/*
 * ============================================================================
 * COMPILATION & EXECUTION FLOW
 * ============================================================================
 * 
 * 1. User runs: python test.py
 * 
 * 2. Python imports cuda_ops
 * 
 * 3. cuda_ops/fusion.py runs torch.utils.cpp_extension.load()
 * 
 * 4. PyTorch calls nvcc to compile this .cu file:
 *    nvcc -c fusion_kernel.cu -o fusion_kernel.o
 * 
 * 5. PyTorch links it into a Python extension:
 *    g++ fusion_kernel.o -o fusion_cuda.so
 * 
 * 6. Python loads fusion_cuda.so
 * 
 * 7. User calls: fused_add_mul_exp(x, y)
 * 
 * 8. Python → fusion.py → _C.fused_add_mul_exp() → fused_add_mul_exp_cuda()
 * 
 * 9. fused_add_mul_exp_cuda() launches kernel on GPU
 * 
 * 10. Kernel executes in parallel on thousands of GPU threads
 * 
 * 11. Result returned to Python
 */
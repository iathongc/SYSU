#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>

// Kernel for matrix multiplication (GEMM)
__global__ void gemmKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    // Input dimensions
    int height = 4096;
    int width = 4096;
    int channels = 3;

    // Kernel dimensions
    int kernel_size = 3;
    int stride = 1;

    // Output dimensions
    int output_height = (height - kernel_size) / stride + 1;
    int output_width = (width - kernel_size) / stride + 1;

    // Host memory allocation
    size_t input_size = channels * height * width * sizeof(float);
    size_t kernel_size_total = channels * kernel_size * kernel_size * sizeof(float);
    size_t output_size = output_height * output_width * sizeof(float);

    float* h_input = (float*)malloc(input_size);
    float* h_kernel = (float*)malloc(kernel_size_total);
    float* h_output = (float*)malloc(output_size);

    // Initialize input and kernel with random values
    for (int i = 0; i < channels * height * width; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < channels * kernel_size * kernel_size; ++i) {
        h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory allocation
    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size_total);
    cudaMalloc(&d_output, output_size);

    // Copy data to device
    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size_total, cudaMemcpyHostToDevice);

    // Configure block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x, (output_height + blockSize.y - 1) / blockSize.y);

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();

    // Launch GEMM kernel
    gemmKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, output_height, output_width, channels * kernel_size * kernel_size);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Copy result back to host
    cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost);

    // Output the first 2x2 results
    std::cout << "Output matrix (first 2*2 elements):" << std::endl;
    for (int i = 0; i < std::min(2, output_height); ++i) {
        for (int j = 0; j < std::min(2, output_width); ++j) {
            std::cout << h_output[i * output_width + j] << " ";
        }
        if (output_width > 2) {
            std::cout << "...";
        }
        std::cout << std::endl;
    }
    if (output_height > 2) {
        std::cout << "..." << std::endl;
    }

    std::cout << "\n";
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    // Free memory
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}

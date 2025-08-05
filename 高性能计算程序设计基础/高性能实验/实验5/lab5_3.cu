#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

#define CHECK_CUDA_CALL(call) {                                 \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl;      \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

__global__ void direct_convolution_2d(
    float* input, float* kernel, float* output,
    int input_h, int input_w, int stride, int padding, int output_h, int output_w) {

    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < output_w && out_y < output_h) {
        float result = 0.0f;

        for (int c = 0; c < 3; ++c) { // Iterate over channels
            for (int ky = 0; ky < 3; ++ky) {
                for (int kx = 0; kx < 3; ++kx) {
                    int in_x = out_x * stride + kx - padding;
                    int in_y = out_y * stride + ky - padding;

                    if (in_x >= 0 && in_x < input_w && in_y >= 0 && in_y < input_h) {
                        result += input[(c * input_h + in_y) * input_w + in_x] *
                                  kernel[(c * 3 + ky) * 3 + kx];
                    }
                }
            }
        }
        output[(out_y * output_w + out_x)] = result;
    }
}

void print_matrix(const float* matrix, int h, int w) {
    // Print the top-left 2x2 block and add ellipses for larger matrices
    for (int i = 0; i < std::min(h, 2); ++i) {
        for (int j = 0; j < std::min(w, 2); ++j) {
            std::cout << matrix[i * w + j] << " ";
        }
        if (w > 2) std::cout << "...";
        std::cout << std::endl;
    }
    if (h > 2) {
        std::cout << "... (truncated)" << std::endl;
    }
}

void run_convolution(int input_h, int input_w, int stride) {
    const int channels = 3;
    const int kernel_size = 3;
    const int padding = (stride - 1) / 2; // Same padding calculation

    int output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    int output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    size_t input_size = input_h * input_w * channels * sizeof(float);
    size_t kernel_size_bytes = kernel_size * kernel_size * channels * sizeof(float);
    size_t output_size = output_h * output_w * sizeof(float);

    float* h_input = (float*)malloc(input_size);
    float* h_kernel = (float*)malloc(kernel_size_bytes);
    float* h_output = (float*)malloc(output_size);

    // Initialize input and kernel with random values
    for (int i = 0; i < input_h * input_w * channels; ++i) h_input[i] = static_cast<float>(rand() % 10 + 1);
    for (int i = 0; i < kernel_size * kernel_size * channels; ++i) h_kernel[i] = static_cast<float>(rand() % 5 + 1);

    float* d_input, * d_kernel, * d_output;
    CHECK_CUDA_CALL(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_CALL(cudaMalloc(&d_kernel, kernel_size_bytes));
    CHECK_CUDA_CALL(cudaMalloc(&d_output, output_size));

    CHECK_CUDA_CALL(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_kernel, h_kernel, kernel_size_bytes, cudaMemcpyHostToDevice));

    dim3 block_dim(16, 16);
    dim3 grid_dim((output_w + block_dim.x - 1) / block_dim.x,
                  (output_h + block_dim.y - 1) / block_dim.y);

    auto start = std::chrono::high_resolution_clock::now(); // Start timer
    direct_convolution_2d<<<grid_dim, block_dim>>>(
        d_input, d_kernel, d_output,
        input_h, input_w, stride, padding, output_h, output_w);
    CHECK_CUDA_CALL(cudaDeviceSynchronize()); // Wait for GPU to finish
    auto end = std::chrono::high_resolution_clock::now(); // End timer

    CHECK_CUDA_CALL(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    // Print input, kernel, and output
    std::cout << "Input Matrix:" << std::endl;
    print_matrix(h_input, input_h, input_w);
    std::cout << "\n";

    std::cout << "Kernel Matrix:" << std::endl;
    print_matrix(h_kernel, kernel_size, kernel_size);
    std::cout << "\n";

    std::cout << "Output Matrix:" << std::endl;
    print_matrix(h_output, output_h, output_w);
    std::cout << "\n";

    // Print runtime
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Convolution runtime: " << elapsed.count() << " seconds" << std::endl;

    // Cleanup
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

int main() {
    int input_h = 4096, input_w = 4096;
    int stride = 1;

    std::cout << "Running convolution with stride = " << stride << std::endl;
    run_convolution(input_h, input_w, stride);

    return 0;
}

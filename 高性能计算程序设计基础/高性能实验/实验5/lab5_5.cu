#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cudnn.h>

// Check CUDA and cuDNN calls for errors
#define CHECK_CUDA(call) {\
    cudaError_t err = call;\
    if (err != cudaSuccess) {\
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

#define CHECK_CUDNN(call) {\
    cudnnStatus_t err = call;\
    if (err != CUDNN_STATUS_SUCCESS) {\
        std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ << " - " << cudnnGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

// Function to print the first 2x2 of a tensor
void printMatrixPreview(const float* matrix, int rows, int cols, const std::string& name) {
    std::cout << name << " (First 2*2):\n";
    for (int i = 0; i < std::min(rows, 2); ++i) {
        for (int j = 0; j < std::min(cols, 2); ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        if (cols > 2)
            std::cout << "...";
        std::cout << "\n";
    }
    if (rows > 2)
        std::cout << "...\n";
}

int main() {
    // Input dimensions: N (batch size), C (channels), H (height), W (width)
    int N = 1, C = 1, H = 5, W = 5;
    int K = 1, R = 3, S = 3; // Output channels, kernel height, kernel width
    int pad_h = 1, pad_w = 1, stride_h = 1, stride_w = 1;

    // Allocate and initialize host memory for input and kernel
    std::vector<float> h_input(N * C * H * W);
    std::vector<float> h_kernel(K * C * R * S);
    std::vector<float> h_output;

    // Initialize input and kernel with some values
    for (int i = 0; i < N * C * H * W; ++i) {
        h_input[i] = static_cast<float>(i + 1); // Example: Sequential values
    }
    for (int i = 0; i < K * C * R * S; ++i) {
        h_kernel[i] = 0.5f; // Example: All elements are 0.5
    }

    // cuDNN handles
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreate(&cudnn));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // Input tensor descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    // Kernel descriptor
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S));

    // Convolution descriptor
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Output dimensions
    int outN, outC, outH, outW;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, &outN, &outC, &outH, &outW));

    h_output.resize(outN * outC * outH * outW);

    // Output tensor descriptor
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outN, outC, outH, outW));

    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, N * C * H * W * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, K * C * R * S * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, outN * outC * outH * outW * sizeof(float)));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel.data(), K * C * R * S * sizeof(float), cudaMemcpyHostToDevice));

    // Convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

    // Workspace allocation
    size_t workspaceSize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));
    void* d_workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));
    }

    // Alpha and beta values
    const float alpha = 1.0f, beta = 0.0f;

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_kernel, convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, d_output));

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end - start;

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, outN * outC * outH * outW * sizeof(float), cudaMemcpyDeviceToHost));

    // Print matrices
    printMatrixPreview(h_input.data(), H, W, "Input");
    std::cout << "\n";
    printMatrixPreview(h_kernel.data(), R, S, "Kernel");
    std::cout << "\n";
    printMatrixPreview(h_output.data(), outH, outW, "Output");

    std::cout << "\n";
    std::cout << "Convolution time: " << elapsed.count() << " ms\n";

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    if (workspaceSize > 0) cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);

    return 0;
}

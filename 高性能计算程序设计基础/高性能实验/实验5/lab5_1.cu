#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath> // 添加 cmath 库以使用 round 函数

#define TILE_SIZE 32 // 线程块维度

// CUDA 核函数进行矩阵乘法
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 打印矩阵（仅打印前 2*2，其他部分用省略号表示）
void printMatrix(float* matrix, int N) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << "... ";
        std::cout << std::endl;
    }
    std::cout << "...\n";
}

int main() {
    const int N = 8192; // 矩阵大小 N x N
    size_t size = N * N * sizeof(float);

    // 分配主机内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // 初始化矩阵 A 和 B
    srand(time(0));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = round((static_cast<float>(rand()) / RAND_MAX) * 1000) / 1000; // 保留 3 位小数
        h_B[i] = round((static_cast<float>(rand()) / RAND_MAX) * 1000) / 1000; // 保留 3 位小数
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 设置 CUDA 网格和块大小
    dim3 dimBlock(TILE_SIZE, TILE_SIZE); // 32 x 32 的线程块，512 个线程/块
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // 记录时间
    auto start = std::chrono::high_resolution_clock::now();

    // 启动 CUDA 核函数
    matrixMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // 同步设备
    cudaDeviceSynchronize();

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;

    // 拷贝结果回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // 打印矩阵
    std::cout << "Matrix A (top-left 2*2):\n";
    printMatrix(h_A, N);
    std::cout <<std::endl;

    std::cout << "Matrix B (top-left 2*2):\n";
    printMatrix(h_B, N);
    std::cout <<std::endl;

    std::cout << "Matrix C (top-left 2*2):\n";
    printMatrix(h_C, N);
    std::cout <<std::endl;

    // 打印执行时间
    std::cout << "Execution time: " << duration.count() << " seconds.\n";

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
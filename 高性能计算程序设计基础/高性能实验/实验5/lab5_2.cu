#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <iomanip>

// 初始化随机矩阵
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f; // 随机数范围 [0, 100]
    }
}

// 打印矩阵首2x2部分
void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < 2 && i < rows; ++i) {
        for (int j = 0; j < 2 && j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(3) << matrix[i * cols + j] << " ";
        }
        if (cols > 2)
            std::cout << "... ";
        std::cout << std::endl;
    }
    if (rows > 2)
        std::cout << "...\n";
}

int main() {
    const int maxSize = 8192;
    const int startSize = 512;

    // 生成最大规模的矩阵 A 和 B
    float* h_A_full = new float[maxSize * maxSize];
    float* h_B_full = new float[maxSize * maxSize];
    initializeMatrix(h_A_full, maxSize, maxSize);
    initializeMatrix(h_B_full, maxSize, maxSize);

    // 打印 A 和 B 的首2*2矩阵
    std::cout << "Matrix A:\n";
    printMatrix(h_A_full, maxSize, maxSize);
    std::cout << "\nMatrix B:\n";
    printMatrix(h_B_full, maxSize, maxSize);

    for (int size = startSize; size <= maxSize; size *= 2) {
        // 分配子矩阵
        float* h_A = h_A_full;
        float* h_B = h_B_full;
        float* h_C = new float[size * size];

        // 设备端内存分配
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size * size * sizeof(float));
        cudaMalloc(&d_B, size * size * sizeof(float));
        cudaMalloc(&d_C, size * size * sizeof(float));

        // 拷贝子矩阵到设备
        cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);

        // 创建 CUBLAS 句柄
        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();

        // 调用 CUBLAS 矩阵相乘
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha,
                    d_A, size, d_B, size, &beta, d_C, size);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // 拷贝结果回主机
        cudaMemcpy(h_C, d_C, size * size * sizeof(float), cudaMemcpyDeviceToHost);

        // 打印结果矩阵 C 的首2*2
        if (size == startSize) {
            std::cout << "\nMatrix C:\n";
            printMatrix(h_C, size, size);
        }

        // 输出运行时间
        std::cout << "\n";
        std::cout << "Execution time (" << size << "x" << size << "): " << elapsed.count() << " seconds\n";

        // 清理设备内存
        delete[] h_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cublasDestroy(handle);
    }

    // 清理主机内存
    delete[] h_A_full;
    delete[] h_B_full;

    return 0;
}

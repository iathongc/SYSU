#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>
#include <cstdlib> 

void initialize_matrix(std::vector<std::vector<double>>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
}

void print_matrix(const std::vector<std::vector<double>>& matrix, int size, bool print_full) {
    if (print_full) {
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                std::cout << matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    else {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                std::cout << matrix[i][j] << " ";
            }
            if (size > 2) 
                std::cout << "..."; // 省略号表示隐藏部分
            std::cout << std::endl;
        }
        if (size > 2) 
            std::cout << "..." << std::endl << std::endl; // 表示矩阵更大
    }
}

// 矩阵乘法函数
void matrix_multiplication(const std::vector<std::vector<double>>& A,
                           const std::vector<std::vector<double>>& B,
                           std::vector<std::vector<double>>& C,
                           int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            double sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    int rows = std::stoi(argv[1]);          // 矩阵的行数
    int cols = std::stoi(argv[2]);          // 矩阵的列数
    int size = std::stoi(argv[3]);          // 矩阵的维数
    int num_threads = std::stoi(argv[4]);   // 使用的线程数
    bool print_matrix_flag = false;          // 在代码里手动控制是否打印矩阵，设为true打印矩阵

    // 初始化矩阵
    std::vector<std::vector<double>> A(size, std::vector<double>(size));
    std::vector<std::vector<double>> B(size, std::vector<double>(size));
    std::vector<std::vector<double>> C(size, std::vector<double>(size, 0.0));

    initialize_matrix(A, size);
    initialize_matrix(B, size);

    // 打印初始矩阵
    if (print_matrix_flag) {
        std::cout << "Matrix A:" << std::endl;
        print_matrix(A, size, false);  // 打印部分矩阵
        std::cout << "Matrix B:" << std::endl;
        print_matrix(B, size, false);  // 打印部分矩阵
    }

    // 矩阵乘法和计时
    auto start_time = std::chrono::high_resolution_clock::now();
    matrix_multiplication(A, B, C, size, num_threads);
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end_time - start_time).count();

    // 打印结果矩阵
    if (print_matrix_flag) {
        std::cout << "Matrix C:" << std::endl;
        print_matrix(C, size, false);  // 打印部分矩阵
    }

    // 输出执行时间
    std::cout << "Execution time: " << duration << " seconds" << std::endl;

    return 0;
}


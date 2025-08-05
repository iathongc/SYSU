#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "parallel_for.h" // 引入 parallel_for 的头文件

// 定义结构体传递矩阵乘法的参数
struct matrix_args {
    const std::vector<std::vector<int>>* A; // 矩阵 A 的指针
    const std::vector<std::vector<int>>* B; // 矩阵 B 的指针
    std::vector<std::vector<int>>* C;       // 矩阵 C 的指针
    int size;                                // 矩阵大小
    int start_row;                           // 当前线程负责的起始行
    int end_row;                             // 当前线程负责的结束行
};

// 被 parallel_for 调用的矩阵乘法函数
void* matrix_multiply_functor(void* args) {
    matrix_args* mat_args = (matrix_args*)args;
    for (int i = mat_args->start_row; i < mat_args->end_row; ++i) {
        for (int j = 0; j < mat_args->size; ++j) {
            int sum = 0;  // 使用整数类型
            for (int k = 0; k < mat_args->size; ++k) {
                sum += (*mat_args->A)[i][k] * (*mat_args->B)[k][j];
            }
            (*mat_args->C)[i][j] = sum;
        }
    }
    return nullptr;
}

// 随机初始化矩阵，元素为整数
void initialize_matrix(std::vector<std::vector<int>>& matrix, int max_value) {
    for (auto& row : matrix) {
        for (auto& elem : row) {
            elem = rand() % (max_value + 1); // 生成 [0, max_value] 范围内的随机整数
        }
    }
}

// 打印矩阵的前 2x2 子矩阵
void print_matrix(const std::vector<std::vector<int>>& matrix) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "..." << std::endl;
    }
    std::cout << "..." << std::endl << std::endl;
}

int main(int argc, char* argv[]) {
    const int size = 1024;      // 矩阵大小
    int num_threads = 1;  // Default thread count

    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
    }
    else {
        std::cout << "Enter the number of threads: ";
        std::cin >> num_threads;
    }

    srand(time(0)); // 初始化随机数种子

    // 初始化矩阵
    std::vector<std::vector<int>> A(size, std::vector<int>(size));
    std::vector<std::vector<int>> B(size, std::vector<int>(size));
    std::vector<std::vector<int>> C(size, std::vector<int>(size, 0));

    initialize_matrix(A, 100);  // 随机生成 [0, 100] 范围的整数
    initialize_matrix(B, 100);  // 随机生成 [0, 100] 范围的整数

    // 输出矩阵 A 和 B
    std::cout << "Matrix A:" << std::endl;
    print_matrix(A);

    std::cout << "Matrix B:" << std::endl;
    print_matrix(B);

    // 开始计时
    auto start_time = std::chrono::high_resolution_clock::now();

    // 分配任务给线程
    int rows_per_thread = size / num_threads;
    std::vector<matrix_args> thread_args(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? size : start_row + rows_per_thread; // 确保最后线程覆盖剩余行

        thread_args[i] = {&A, &B, &C, size, start_row, end_row};
    }

    // 使用 parallel_for 调用
    for (int i = 0; i < num_threads; ++i) {
        parallel_for(i, i + 1, 1, matrix_multiply_functor, &thread_args[i], num_threads);
    }

    // 结束计时
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end_time - start_time).count();

    // 输出矩阵 C和执行时间
    std::cout << "Matrix C:" << std::endl;
    print_matrix(C);

    std::cout << "Execution time: " << duration << " seconds" << std::endl;

    return 0;
}

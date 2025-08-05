#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <chrono>

// 矩阵初始化函数
void initialize_matrix(std::vector<std::vector<double>>& matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 10.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
}

// 打印矩阵（前2x2部分，其余用省略号表示）
void print_matrix(const std::vector<std::vector<double>>& matrix, int rows, int cols) {
    for (int i = 0; i < std::min(rows, 2); ++i) {
        for (int j = 0; j < std::min(cols, 2); ++j) {
            std::cout << matrix[i][j] << " ";
        }
        if (cols > 2) std::cout << "...";
            std::cout << std::endl;
    }
    if (rows > 2)
        std::cout << "...\n" << std::endl;
}

// 矩阵乘法函数
void matrix_multiplication(const std::vector<std::vector<double>>& A,
                           const std::vector<std::vector<double>>& B,
                           std::vector<std::vector<double>>& C,
                           int M, int N, int K, int num_threads, int schedule_type) {
    switch (schedule_type) {
        case 1:
            omp_set_schedule(omp_sched_auto, 1);
            break;
        case 2:
            omp_set_schedule(omp_sched_static, 1);
            break;
        case 3:
            omp_set_schedule(omp_sched_dynamic, 1);
            break;
        default:
            std::cerr << "Invalid schedule type. Using default scheduling.\n";
            omp_set_schedule(omp_sched_auto, 1);
            break;
    }

    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K> <num_threads> <schedule_type>\n";
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    int num_threads = std::stoi(argv[4]);
    int schedule_type = std::stoi(argv[5]);

    // 控制是否打印矩阵的变量
    bool print_flag = false;

    // 初始化矩阵
    std::vector<std::vector<double>> A(M, std::vector<double>(N));
    std::vector<std::vector<double>> B(N, std::vector<double>(K));
    std::vector<std::vector<double>> C(M, std::vector<double>(K, 0.0));

    initialize_matrix(A, M, N);
    initialize_matrix(B, N, K);

    if (print_flag) {
        std::cout << "Matrix A:\n";
        print_matrix(A, M, N);
        std::cout << "Matrix B:\n";
        print_matrix(B, N, K);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    matrix_multiplication(A, B, C, M, N, K, num_threads, schedule_type);
    auto end_time = std::chrono::high_resolution_clock::now();

    if (print_flag) {
        std::cout << "Matrix C (Result of A * B):\n";
        print_matrix(C, M, K);
    }

    double duration = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "Execution time: " << duration << " seconds\n";

    return 0;
}


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1200
#define N 1000
#define K 800

void generate_random_matrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (float)rand() / RAND_MAX;
        }
    }
}

void multiply_matrices(float** A, float** B, float** C, int m, int n, int k) {
    for (int j = 0; j < k; j++) {
        for (int l = 0; l < n; l++) {
            for (int i = 0; i < m; i++) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
}

void print_matrix(float** matrix, int rows, int cols) {
    int display_rows = (rows < 2) ? rows : 2;
    int display_cols = (cols < 2) ? cols : 2;

    for (int i = 0; i < display_rows; i++) {
        for (int j = 0; j < display_cols; j++) {
            printf("%f ", matrix[i][j]);
        }
        if (cols > 2) {
            printf("...");
        }
        printf(" \n");
    }
    if (rows > 2) {
        printf("...\n");
    }
}

float** allocate_matrix(int rows, int cols) {
    float** matrix = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (float*)malloc(cols * sizeof(float));
    }
    return matrix;
}

void free_matrix(float** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    // 动态分配内存
    float** A = allocate_matrix(M, N);
    float** B = allocate_matrix(N, K);
    float** C = allocate_matrix(M, K);

    srand((unsigned int)time(NULL));

    // 生成随机矩阵 A 和 B
    generate_random_matrix(A, M, N);
    generate_random_matrix(B, N, K);

    printf("Calculating running time...\n");
    // 记录开始时间
    clock_t start_time = clock();

    // 进行矩阵乘法
    multiply_matrices(A, B, C, M, N, K);

    // 记录结束时间
    clock_t end_time = clock();
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // 打印矩阵 A, B, 和 C（只打印前 2*2 个元素，并使用省略号代替超出部分）
    printf("\nMatrix A (first 2*2 elements):\n");
    print_matrix(A, M, N);

    printf("\nMatrix B (first 2*2 elements):\n");
    print_matrix(B, N, K);

    printf("\nMatrix C (Result of A*B, first 2*2 elements):\n");
    print_matrix(C, M, K);

    // 打印执行时间
    printf("\nRun time: %f seconds.\n", time_taken);
    printf("\n");

    // 释放内存
    free_matrix(A, M);
    free_matrix(B, N);
    free_matrix(C, M);

    return 0;
}

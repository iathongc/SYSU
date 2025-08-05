#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define M 500
#define N 500
#define EPSILON 0.001

double u[M][N], w[M][N];

int main(int argc, char *argv[]) {
    int rank, size;
    int i, j, iterations = 0, iterations_print = 1;
    double mean = 0.0, diff = EPSILON, local_diff, global_diff, start_time, end_time;

    // 初始化 MPI 环境
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 每个进程分配的行范围
    int rows_per_proc = M / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank == size - 1) ? M - 2 : start_row + rows_per_proc - 1;

    // 只有主进程初始化边界条件
    if (rank == 0) {
        for (i = 0; i < M; i++) {
            w[i][0] = 100.0;
            w[i][N - 1] = 100.0;
        }
        for (j = 0; j < N; j++) {
            w[0][j] = 0.0;
            w[M - 1][j] = 100.0;
        }

        for (i = 1; i < M - 1; i++) {
            mean += w[i][0] + w[i][N - 1];
        }
        for (j = 0; j < N; j++) {
            mean += w[0][j] + w[M - 1][j];
        }
        mean /= (2 * M + 2 * N - 4);
    }

    // 广播边界值和初始平均值
    MPI_Bcast(&w, M * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 初始化内部点
    for (i = start_row; i <= end_row; i++) {
        for (j = 1; j < N - 1; j++) {
            w[i][j] = mean;
        }
    }

    // 主进程打印信息
    if (rank == 0) {
        printf("HEATED_PLATE_MPI\n");
        printf("  Grid size: %d x %d\n", M, N);
        printf("  Number of processes: %d\n", size);
        printf("  Iteration until diff <= %e\n", EPSILON);
        printf("\n Iteration  Change\n");
    }

    start_time = MPI_Wtime();

    // 迭代计算
    while (diff >= EPSILON) {
        // 备份解
        memcpy(u, w, sizeof(w));

        // 更新内部点
        for (i = start_row; i <= end_row; i++) {
            for (j = 1; j < N - 1; j++) {
                w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;
            }
        }

        // 边界数据交换
        if (rank > 0) {
            MPI_Send(&w[start_row][1], N - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&w[start_row - 1][1], N - 2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(&w[end_row][1], N - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&w[end_row + 1][1], N - 2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // 计算局部最大差值
        local_diff = 0.0;
        for (i = start_row; i <= end_row; i++) {
            for (j = 1; j < N - 1; j++) {
                double temp_diff = fabs(w[i][j] - u[i][j]);
                if (temp_diff > local_diff) {
                    local_diff = temp_diff;
                }
            }
        }

        // 归约计算全局最大差值
        MPI_Allreduce(&local_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // 主进程打印迭代信息
        if (rank == 0) {
            iterations++;
            if (iterations == iterations_print) {
                printf("  %8d  %f\n", iterations, diff);
                iterations_print *= 2;
            }
        }
    }

    end_time = MPI_Wtime();

    // 主进程打印最终结果
    if (rank == 0) {
        printf("\n  %8d  %f\n", iterations, diff);
        printf("\n  Error tolerance achieved.\n");
        printf("  Wallclock time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}


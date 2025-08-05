#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define M 500
#define N 500
#define NUM_THREADS 4

typedef struct {
    int start, end;
    void (*func)(int, void*);
    void* arg;
} ThreadData;

void* thread_work(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start; i < data->end; ++i) {
        data->func(i, data->arg);
    }
    return NULL;
}

void parallel_for(int start, int end, void (*func)(int, void*), void* arg) {
    pthread_t threads[NUM_THREADS];
    ThreadData thread_data[NUM_THREADS];
    int chunk_size = (end - start) / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; ++t) {
        thread_data[t].start = start + t * chunk_size;
        thread_data[t].end = (t == NUM_THREADS - 1) ? end : start + (t + 1) * chunk_size;
        thread_data[t].func = func;
        thread_data[t].arg = arg;
        pthread_create(&threads[t], NULL, thread_work, &thread_data[t]);
    }

    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_join(threads[t], NULL);
    }
}

// 全局变量
double u[M][N];
double w[M][N];
double diff;
pthread_mutex_t diff_mutex = PTHREAD_MUTEX_INITIALIZER;

// 更新解
void update_solution(int i, void* arg) {
    (void)arg; // 无用参数，为了与接口匹配
    for (int j = 1; j < N - 1; ++j) {
        w[i][j] = (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1]) / 4.0;
    }
}

// 主函数
int main() {
    double epsilon = 0.001;
    double mean = 0.0;
    int iterations = 0;
    int iterations_print = 1;

    printf("\n");
    printf("HEATED_PLATE_PTHREADS\n");
    printf("  Pthreads version\n");
    printf("  A program to solve for the steady state temperature distribution\n");
    printf("  over a rectangular plate.\n");
    printf("\n");
    printf("  Spatial grid of %d by %d points.\n", M, N);
    printf("  The iteration will be repeated until the change is <= %e\n", epsilon);

    // 初始化边界条件
    for (int i = 1; i < M - 1; ++i) {
        w[i][0] = 100.0;
        w[i][N - 1] = 100.0;
    }
    for (int j = 0; j < N; ++j) {
        w[M - 1][j] = 100.0;
        w[0][j] = 0.0;
    }

    // 计算边界平均值
    for (int i = 1; i < M - 1; ++i) {
        mean += w[i][0] + w[i][N - 1];
    }
    for (int j = 0; j < N; ++j) {
        mean += w[M - 1][j] + w[0][j];
    }

    mean /= (2 * M + 2 * N - 4);
    printf("\n  MEAN = %f\n", mean);

    // 初始化内部区域
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            w[i][j] = mean;
        }
    }

    printf("\n  Iteration  Change\n\n");

    clock_t start_time = clock(); // 记录开始时间

    // 迭代计算
    do {
        diff = 0.0;

        // 备份当前解
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                u[i][j] = w[i][j];
            }
        }

        // 并行更新解
        parallel_for(1, M - 1, update_solution, NULL);

        // 计算最大变化值
        for (int i = 1; i < M - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                double temp_diff = fabs(w[i][j] - u[i][j]);
                pthread_mutex_lock(&diff_mutex);
                if (temp_diff > diff) {
                    diff = temp_diff;
                }
                pthread_mutex_unlock(&diff_mutex);
            }
        }

        iterations++;
        if (iterations == iterations_print) {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print *= 2;
        }
    } while (diff > epsilon);

    clock_t end_time = clock(); // 记录结束时间
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\n  %8d  %f\n", iterations, diff);
    printf("\n  Error tolerance achieved.\n");
    printf("  Wallclock time = %f seconds\n", elapsed_time);

    printf("\nHEATED_PLATE_PTHREADS:\n");
    printf("  Normal end of execution.\n");

    return 0;
}


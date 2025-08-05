#include "parallel_for.h"
#include <pthread.h>

// 线程函数
void* thread_function(void* args) {
    struct for_index* index = (struct for_index*)args;
    return index->functor(index->arg); // 调用 functor
}

// parallel_for实现
void parallel_for(int start, int end, int increment, void* (*functor)(void*), void* arg, int num_threads) {
    // 创建线程
    pthread_t threads[num_threads];
    struct for_index thread_args[num_threads];

    // 计算任务分配
    int chunk_size = (end - start) / num_threads;
    int remainder = (end - start) % num_threads; // 处理不能整除的情况

    for (int i = 0; i < num_threads; ++i) {
        thread_args[i].start = start + i * chunk_size + (i < remainder ? i : remainder);
        thread_args[i].end = thread_args[i].start + chunk_size + (i < remainder ? 1 : 0);
        thread_args[i].increment = increment;
        thread_args[i].functor = functor;
        thread_args[i].arg = arg;

        pthread_create(&threads[i], nullptr, thread_function, &thread_args[i]);
    }

    // 等待所有线程完成
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }
}


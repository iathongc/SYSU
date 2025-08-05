#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#include <pthread.h>

// 用于线程分工的结构体
struct for_index {
    int start;                  // 起始索引
    int end;                    // 结束索引
    int increment;              // 增量
    void* (*functor)(void*);    // 函数指针，线程要执行的任务
    void* arg;                  // functor 的参数
};

// parallel_for 函数声明
void parallel_for(int start, int end, int increment, void* (*functor)(void*), void* arg, int num_threads);

#endif // PARALLEL_FOR_H


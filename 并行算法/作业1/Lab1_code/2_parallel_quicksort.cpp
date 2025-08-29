/*************************************************
 * 文件名：2_parallel_quicksort.cpp
 * 姓名：陳日康
 * 描述：使用 std::thread 实现并行 QuickSort算法，
 *       支持多线程递归排序。
 *************************************************/

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <mutex>
#include <map>
#include <set>

int MAX_THREAD_DEPTH = 2;  // 控制递归并行的最大深度

std::mutex print_mutex;  // 输出线程编号的互斥锁
std::map<std::thread::id, int> thread_number_map;  // 逻辑线程编号映射表
int thread_counter = 0;
std::set<std::thread::id> printed_threads;  // 已经输出过编号的线程集合

/*************************************************
 * 函数名：printLogicalThreadIdOnce
 * 功能描述：每个线程仅输出一次其编号，便于观察线程调度
 *************************************************/
void printLogicalThreadIdOnce() {
    std::thread::id tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(print_mutex);
    if (!printed_threads.count(tid)) {
        printed_threads.insert(tid);
        if (!thread_number_map.count(tid)) {
            thread_number_map[tid] = thread_counter++;
        }
        std::cout << "Thread " << thread_number_map[tid] << " executed.\n";
    }
}

/*************************************************
 * 函数名：partition
 * 功能描述：对数组进行一次快排划分，返回分割点
 * 参数说明：
 *   arr - 待排序数组
 *   low - 起始索引
 *   high - 结束索引
 *************************************************/
int partition(std::vector<int>& arr, int low, int high) {
    int pivot = arr[high], i = low-1;
    for (int j=low; j<high; ++j) {
        if (arr[j] < pivot)
            std::swap(arr[++i], arr[j]);
    }
    std::swap(arr[i+1], arr[high]);
    return i+1;
}

/*************************************************
 * 函数名：quickSort
 * 功能描述：递归实现并行快速排序，线程数由深度控制
 * 参数说明：
 *   arr - 待排序数组
 *   low - 起始索引
 *   high - 结束索引
 *   depth - 当前递归深度
 *************************************************/
void quickSort(std::vector<int>& arr, int low, int high, int depth = 0) {
    printLogicalThreadIdOnce();  // 每个线程仅输出一次编号

    if (low<high) {
        int pi = partition(arr, low, high);  // 获取分区位置

        if (depth < MAX_THREAD_DEPTH) {
            // 并行处理左半部分
            std::thread t1([&arr, low, pi, depth]() {
                quickSort(arr, low, pi-1, depth+1);
            });

            // 当前线程处理右半部分
            quickSort(arr, pi+1, high, depth+1);
            t1.join();  // 等待子线程结束
        }
        else {
            // 超过最大深度，转为串行递归
            quickSort(arr, low, pi-1, depth+1);
            quickSort(arr, pi+1, high, depth+1);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc<3) {
        std::cerr << "Usage: ./2_parallel_quicksort <data_size> <threads>\n";
        return 1;
    }

    int n = std::stoi(argv[1]);         // 获取数据规模
    int threads = std::stoi(argv[2]);   // 获取线程数量

    // 计算递归最大深度（控制线程总数）
    int depth = 0;
    while ((1<<depth) < threads) depth++;
    MAX_THREAD_DEPTH = depth;

    std::vector<int> arr(n);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 1e6);
    for (auto& x : arr)
        x = dist(rng);  // 填充随机数据

    auto start = std::chrono::high_resolution_clock::now();
    quickSort(arr, 0, n-1);  // 执行排序
    auto end = std::chrono::high_resolution_clock::now();

    // 输出耗时
    std::cout << "Execution time: " << std::chrono::duration<double>(end - start).count() << "s\n";

    // 验证排序是否成功
    if (std::is_sorted(arr.begin(), arr.end()))
        std::cout << "Sorted correctly.\n";
    else
        std::cout << "Sort failed.\n";

    return 0;
}


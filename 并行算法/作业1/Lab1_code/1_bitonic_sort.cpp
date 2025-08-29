/*************************************************
 * 文件名：1_bitonic_sort.cpp
 * 姓名：陳日康
 * 描述：使用 std::thread 实现并行 Bitonic Sort算法，
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
#include <cmath>

int MAX_THREAD_DEPTH = 2;  // 最大线程深度，控制线程生成层级
std::mutex print_mutex;   // 用于线程安全打印
std::map<std::thread::id, int> thread_id_map;  // 记录每个线程的编号
int thread_counter = 0;   // 分配给线程的逻辑编号计数器

/*************************************************
 * 函数名：printThreadOnce
 * 功能描述：每个线程只输出一次其逻辑编号
 *************************************************/
void printThreadOnce() {
    std::thread::id tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(print_mutex);
    if (!thread_id_map.count(tid)) {
        thread_id_map[tid] = thread_counter++;
        std::cout << "Thread " << thread_id_map[tid] << " executed.\n";
    }
}

/*************************************************
 * 函数名：bitonicMerge
 * 功能描述：对数组的某一段进行 bitonic 合并操作
 * 参数说明：
 *   arr - 待排序数组
 *   low - 起始索引
 *   cnt - 合并长度
 *   dir - 排序方向（true 表示升序）
 *************************************************/
void bitonicMerge(std::vector<int>& arr, int low, int cnt, bool dir) {
    if (cnt>1) {
        int k = cnt/2;  // 拆分成两个部分进行比较
        for (int i=low; i<low+k; ++i) {
            // 根据排序方向交换数据
            if (dir == (arr[i] > arr[i+k])) {
                std::swap(arr[i], arr[i+k]);
            }
        }
        bitonicMerge(arr, low, k, dir);          // 递归合并前半段
        bitonicMerge(arr, low + k, k, dir);      // 递归合并后半段
    }
}

/*************************************************
 * 函数名：bitonicSort
 * 功能描述：递归实现 bitonic sort，支持多线程并行
 * 参数说明：
 *   arr - 待排序数组
 *   low - 起始索引
 *   cnt - 排序长度
 *   dir - 排序方向
 *   depth - 当前递归深度（用于控制线程数量）
 *************************************************/
void bitonicSort(std::vector<int>& arr, int low, int cnt, bool dir, int depth = 0) {
    printThreadOnce();

    if (cnt>1) {
        int k = cnt/2;  // 将数组分成两段

        if (depth < MAX_THREAD_DEPTH) {
            // 在可用线程范围内，新开线程排序左半段
            std::thread t1(bitonicSort, std::ref(arr), low, k, true, depth+1);
            bitonicSort(arr, low+k, k, false, depth+1);  // 当前线程处理右半段
            t1.join();  // 等待子线程完成
        }
        else {
            // 递归调用
            bitonicSort(arr, low, k, true, depth+1);
            bitonicSort(arr, low+k, k, false, depth+1);
        }

        // 合并排序结果
        bitonicMerge(arr, low, cnt, dir);
    }
}

int main(int argc, char* argv[]) {
    if (argc<3) {
        std::cerr << "Usage: ./2_parallel_quicksort <data_size> <threads>\n";
        return 1;
    }
    
    int n = std::stoi(argv[1]);         // 从命令行获取数据规模
    int threads = std::stoi(argv[2]);   // 从命令行获取线程数

    // 检查数据规模是否为2的幂
    if ((n&(n-1)) != 0) {
        std::cerr << "Error: data size must be a power of 2.\n";
        return 1;
    }

    std::vector<int> arr(n);  // 初始化数组

    // 使用随机数生成器填充数组
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(1, 1e6);
    for (int i=0; i<n; ++i)
        arr[i] = dist(rng);

    // 根据线程数计算最大线程深度
    int depth = 0;
    while ((1<<depth) < threads)
        depth++;
    MAX_THREAD_DEPTH = depth;

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();
    bitonicSort(arr, 0, n, true);  // 调用并行排序函数
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Execution time: " << std::chrono::duration<double>(end-start).count() << "s\n";

    // 验证排序结果是否正确
    if (std::is_sorted(arr.begin(), arr.end()))
        std::cout << "Sorted correctly.\n";
    else
        std::cout << "Sort failed.\n";

    return 0;
}


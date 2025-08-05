#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <mutex>
#include <cstdlib> // 用于 std::atoi

std::mutex mtx;
long long totalPoints = 0;
long long insideCurvePoints = 0;

void monteCarloEstimation(int numPoints) {
    // 使用随机数生成器
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    long long localInsideCurvePoints = 0;

    for (int i = 0; i < numPoints; ++i) {
        double x = distribution(rng);
        double y = distribution(rng);

        // 检查点 (x, y) 是否在 y = x^2 曲线下方
        if (y <= x * x) {
            localInsideCurvePoints++;
        }
    }

    // 线程安全地更新全局变量
    {
        std::lock_guard<std::mutex> lock(mtx);
        insideCurvePoints += localInsideCurvePoints;
        totalPoints += numPoints;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <number_of_threads> <points_per_thread>\n";
        return 1;
    }

    int numThreads = std::atoi(argv[1]);
    int pointsPerThread = std::atoi(argv[2]);

    if (numThreads <= 0 || pointsPerThread <= 0) {
        std::cerr << "Error: Number of threads and points per thread must be positive integers.\n";
        return 1;
    }

    std::vector<std::thread> threads;

    // 创建线程
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(monteCarloEstimation, pointsPerThread);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    // 估算面积
    double area = static_cast<double>(insideCurvePoints) / totalPoints;

    // 输出结果
    std::cout << "Estimated area: " << area << std::endl;

    return 0;
}


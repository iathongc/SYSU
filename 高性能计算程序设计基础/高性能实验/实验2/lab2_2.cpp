#include <iostream>
#include <thread>
#include <condition_variable>
#include <cmath>
#include <vector>

// 定义全局变量和条件变量
std::mutex mtx;
std::condition_variable cv;
int completedThreads = 0;
std::vector<double> roots(2, NAN);

void calculateRoot1(double a, double b, double discriminant) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (discriminant > 0) {
            roots[0] = (-b + sqrt(discriminant)) / (2 * a);
        }
        else if (discriminant == 0) {
            roots[0] = -b / (2 * a);
        }
        completedThreads++;
        cv.notify_one();
    }
}

void calculateRoot2(double a, double b, double discriminant) {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (discriminant > 0) {
            roots[1] = (-b - sqrt(discriminant)) / (2 * a);
        }
        else if (discriminant == 0) {
            roots[1] = -b / (2 * a);
        }
        completedThreads++;
        cv.notify_one();
    }
}

void waitForCompletion() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return completedThreads == 2; });
}

int main() {
    double a, b, c;
    std::cout << "Enter the coefficients of the equation: ";
    std::cin >> a >> b >> c;

    double discriminant = b * b - 4 * a * c;

    // 创建线程来计算两个根
    std::thread t1(calculateRoot1, a, b, discriminant);
    std::thread t2(calculateRoot2, a, b, discriminant);

    // 主线程等待所有计算线程完成
    waitForCompletion();

    // 输出结果
    if (discriminant > 0) {
        std::cout << "The two roots of the equation are: " << roots[0] << " and " << roots[1] << std::endl;
    }
    else if (discriminant == 0) {
        std::cout << "The equation has one real root: " << roots[0] << std::endl;
    }
    else {
        std::cout << "The equation has no real roots." << std::endl;
    }

    // 线程回收
    t1.join();
    t2.join();

    return 0;
}


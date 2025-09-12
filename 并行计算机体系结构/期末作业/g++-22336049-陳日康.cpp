#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>
#include <cstring>  // For memset
#include <omp.h>  // For OpenMP

class FigureProcessor {
private:
  unsigned char* figure;
  unsigned char* result;
  const size_t size;

  // Helper to calculate 1D index from 2D indices
  size_t index(size_t i, size_t j) const { return i * size + j; }

public:
  FigureProcessor(size_t size, size_t seed = 0) : size(size) {
    // Allocate memory for the figure and result arrays
    figure = (unsigned char*)calloc(size * size, sizeof(unsigned char));  // calloc initializes memory to 0
    result = (unsigned char*)calloc(size * size, sizeof(unsigned char));

    if (!figure || !result) {
      throw std::bad_alloc();  // Handle allocation failure
    }

    // Random number generation for figure initialization
    // 如果你需要修改内存的数据结构，请不要修改初始化的顺序和逻辑
    // 助教可能会通过指定某个初始化seed 的方式来验证你的代码
    // 如果你修改了初始化的顺序，可能会导致你的代码无法通过测试
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // Fill the figure array with random values
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        figure[index(i, j)] = static_cast<unsigned char>(distribution(gen));
      }
    }
  }

  ~FigureProcessor() {
    free(figure);
    free(result);
  }

  // Gaussian filter (parallelized with OpenMP)
  void gaussianFilter() {
    #pragma omp parallel for collapse(2)  // Parallelize the outer two loops
    for (size_t i = 1; i < size - 1; ++i) {
      for (size_t j = 1; j < size - 1; ++j) {
        result[index(i, j)] =
            (figure[index(i - 1, j - 1)] + 2 * figure[index(i - 1, j)] +
             figure[index(i - 1, j + 1)] + 2 * figure[index(i, j - 1)] +
             4 * figure[index(i, j)] + 2 * figure[index(i, j + 1)] +
             figure[index(i + 1, j - 1)] + 2 * figure[index(i + 1, j)] +
             figure[index(i + 1, j + 1)]) / 16;
      }
    }

    // Handle left and right edges (can be parallelized as well)
    #pragma omp parallel for
    for (size_t i = 1; i < size - 1; ++i) {
      result[index(i, 0)] =
          (figure[index(i - 1, 0)] + 2 * figure[index(i - 1, 0)] +
           figure[index(i - 1, 1)] + 2 * figure[index(i, 0)] +
           4 * figure[index(i, 0)] + 2 * figure[index(i, 1)] +
           figure[index(i + 1, 0)] + 2 * figure[index(i + 1, 0)] +
           figure[index(i + 1, 1)]) / 16;

      result[index(i, size - 1)] =
          (figure[index(i - 1, size - 2)] + 2 * figure[index(i - 1, size - 1)] +
           figure[index(i - 1, size - 1)] + 2 * figure[index(i, size - 2)] +
           4 * figure[index(i, size - 1)] + 2 * figure[index(i, size - 1)] +
           figure[index(i + 1, size - 2)] + 2 * figure[index(i + 1, size - 1)] +
           figure[index(i + 1, size - 1)]) / 16;
    }

    // Handle top and bottom edges (can be parallelized as well)
    #pragma omp parallel for
    for (size_t j = 1; j < size - 1; ++j) {
      result[index(0, j)] =
          (figure[index(0, j - 1)] + 2 * figure[index(0, j)] +
           figure[index(0, j + 1)] + 2 * figure[index(0, j - 1)] +
           4 * figure[index(0, j)] + 2 * figure[index(0, j + 1)] +
           figure[index(1, j - 1)] + 2 * figure[index(1, j)] +
           figure[index(1, j + 1)]) / 16;

      result[index(size - 1, j)] =
          (figure[index(size - 2, j - 1)] + 2 * figure[index(size - 2, j)] +
           figure[index(size - 2, j + 1)] + 2 * figure[index(size - 1, j - 1)] +
           4 * figure[index(size - 1, j)] + 2 * figure[index(size - 1, j + 1)] +
           figure[index(size - 1, j - 1)] + 2 * figure[index(size - 1, j)] +
           figure[index(size - 1, j + 1)]) / 16;
    }

    // Handle corners (can be parallelized as well)
    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) {
      if (i == 0) {
        result[index(0, 0)] = (4 * figure[index(0, 0)] + 2 * figure[index(0, 1)] +
                               2 * figure[index(1, 0)] + figure[index(1, 1)]) / 9;
      }
      else if (i == 1) {
        result[index(0, size - 1)] = (4 * figure[index(0, size - 1)] + 2 * figure[index(0, size - 2)] +
                                       2 * figure[index(1, size - 1)] + figure[index(1, size - 2)]) / 9;
      }
      else if (i == 2) {
        result[index(size - 1, 0)] = (4 * figure[index(size - 1, 0)] + 2 * figure[index(size - 1, 1)] +
                                       2 * figure[index(size - 2, 0)] + figure[index(size - 2, 1)]) / 9;
      }
      else {
        result[index(size - 1, size - 1)] = (4 * figure[index(size - 1, size - 1)] +
                                             2 * figure[index(size - 1, size - 2)] +
                                             2 * figure[index(size - 2, size - 1)] +
                                             figure[index(size - 2, size - 2)]) / 9;
      }
    }
  }

  // Power law transformation using LUT (parallelized with OpenMP)
  void powerLawTransformation() {
    constexpr float gamma = 0.5f;
    float lut[256];
    for (int i = 0; i < 256; ++i) {
      lut[i] = 255.0f * std::pow(i / 255.0f, gamma) + 0.5f;
    }

    #pragma omp parallel for collapse(2)  // Parallelize both loops
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        result[index(i, j)] = static_cast<unsigned char>(lut[figure[index(i, j)]]);
      }
    }
  }

  // Checksum calculation
  unsigned int calcChecksum() {
    unsigned int sum = 0;
    constexpr size_t mod = 1000000007;
    for (size_t i = 0; i < size * size; ++i) {
      sum += result[i];
      sum %= mod;
    }
    return sum;
  }

  // Run benchmark
  void runBenchmark() {
    auto start = std::chrono::high_resolution_clock::now();
    gaussianFilter();
    auto middle = std::chrono::high_resolution_clock::now();

    unsigned int sum = calcChecksum();

    auto middle2 = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    auto end = std::chrono::high_resolution_clock::now();

    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";

    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(middle - start) +
        std::chrono::duration_cast<std::chrono::milliseconds>(end - middle2);
    std::cout << "Benchmark time: " << milliseconds.count() << " ms\n";
  }
};

// Main function
// !!! Please do not modify the main function !!!
int main(int argc, const char** argv) {
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}

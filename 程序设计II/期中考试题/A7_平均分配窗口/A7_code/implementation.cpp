#include "implementation.h"
#include <cmath>
#include <iostream>

// 析构函数释放动态内存
Implementation::~Implementation() {
    for (int i=0; i<lineCount; ++i) {
        delete[] val[i];
    }
    delete[] val;
    delete[] lineCapacity;
    delete[] valCount;
}

// 构造新的一行，如果容量不足就扩容
void Implementation::constructNewLine() {
    if (lineCount == capacity) {
        capacity *= 2;
        int **newVal = new int *[capacity];
        int *newLineCapacity = new int[capacity];
        int *newValCount = new int[capacity]; 
        
        for (size_t i=0; i<lineCount; ++i) {
            newVal[i] = val[i];
            newLineCapacity[i] = lineCapacity[i];
            newValCount[i] = valCount[i];
        } 
        delete[] val;
        delete[] lineCapacity;
        delete[] valCount;    
        val = newVal;
        lineCapacity = newLineCapacity;
        valCount = newValCount;
    }   
    val[lineCount] = new int[NormalSize];
    lineCapacity[lineCount] = NormalSize;
    valCount[lineCount] = 0;
    lineCount++;
}

// 插入一个值到当前最后一行
void Implementation::insert(int value) {
    int last = lineCount - 1;
    if (valCount[last] == lineCapacity[last]) {
        lineCapacity[last] *= 2;
        int *newRow = new int[lineCapacity[last]];
        for (int i=0; i<valCount[last]; ++i) {
            newRow[i] = val[last][i];
        }
        delete[] val[last];
        val[last] = newRow;
    }
    val[last][valCount[last]++] = value;
}

// 字符串转整数
int Implementation::stringToInt(const std::string &input) {
    return std::stoi(input);
}

// 判断素数
bool isPrime(int val) {
    if (val<=1)
        return false;
    for (int i=2; i<=std::sqrt(val); ++i) {
        if (val%i == 0)
            return false;
    }
    return true;
}

// 初始化结果数组
int **initialResult(int n, int k) { 
    int **result = new int *[n];
    for (int i=0; i<n; ++i) {
        result[i] = new int[k];
    }
    return result;
}

// 正确的轮转穿插分配逻辑
void process(Implementation &imp, int n, int k, int **result) {
    int *count = new int[n](); // 每个窗口当前人数
    int startWindow = 0;
    int filled = 0;
    int total = n*k;
    int lineCount = imp.getLineCount();
    int *pos = new int[lineCount](); // 每个队伍当前读取位置

    while (filled < total) {
      for (int l=0; l<lineCount && filled<total; ++l) {
            int added = 0;
            for (int j=pos[l]; j<imp.getLineSize(l); ++j) {
                int val = imp.getVal(l, j);
                if (!isPrime(val))
                    continue;
                while (count[startWindow] == k) {
                    startWindow = (startWindow+1) % n;
                }
                result[startWindow][count[startWindow]++] = val;
                filled++;
                startWindow = (startWindow+1) % n;
                added++;
                pos[l] = j+1;
                if (added == n || filled == total)
                    break;
            }
        }
    }
    delete[] count;
    delete[] pos;
}
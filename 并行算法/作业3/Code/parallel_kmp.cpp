/************************************************************
 * 文件名称：parallel_kmp.cpp
 * 姓    名：陳日康
 * 学    号：22336049
 * 实验目的：并行实现 KMP 字符串匹配算法，比较多线程加速效果
 ************************************************************/
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
using namespace std;

/************************************************************
 * 函数名称: generate_random_string
 * 函数功能: 生成由小写字母构成的随机字符串
 * 输入参数: len 要生成的字符串长度
 * 返回结果: 随机字符串
 ************************************************************/
string generate_random_string(int len) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyz";
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 25);
    string s;
    s.reserve(len);
    for (int i=0; i<len; ++i)
        s += charset[dis(gen)];
    return s;
}

/************************************************************
 * 函数名称: insert_pattern_at_positions
 * 函数功能: 将模式串插入到主串的指定位置
 * 输入参数: text主串；pattern模式串；positions插入位置列表
 ************************************************************/
void insert_pattern_at_positions(string& text, const string& pattern, const vector<int>& positions) {
    for (int pos:positions) {
        if (pos+pattern.size() <= text.size())
            copy(pattern.begin(), pattern.end(), text.begin() + pos);
    }
}

/************************************************************
 * 函数名称: build_kmp_table
 * 函数功能: 构建 KMP 算法中的前缀函数表
 * 输入参数: pattern 模式串
 * 返回结果: KMP 前缀数组
 ************************************************************/
vector<int> build_kmp_table(const string& pattern) {
    int m = pattern.size();
    vector<int> prefix(m, 0);
    for (int i=1, j=0; i<m;) {
        if (pattern[i] == pattern[j])
            prefix[i++] = ++j;
        else if (j>0)
            j = prefix[j-1];
        else
            prefix[i++] = 0;
    }
    return prefix;
}

/************************************************************
 * 函数名称：kmp_search
 * 函数功能：执行 KMP 匹配，找出所有匹配位置
 * 输入参数：text主串；pattern模式串；prefix前缀表；offset起始偏移
 * 返回结果：所有匹配位置（全局坐标）
 ************************************************************/
vector<int> kmp_search(const string& text, const string& pattern, const vector<int>& prefix, int offset=0) {
    vector<int> result;
    int n=text.size(), m=pattern.size();
    for (int i=0, j=0; i<n;) {
        if (text[i] == pattern[j]) {
            ++i;
            ++j;
            if (j==m) {
                result.push_back(i-j+offset);
                j = prefix[j-1];
            }
        }
        else if (j>0)
            j = prefix[j-1];
        else
            ++i;
    }
    return result;
}

/************************************************************
 * 函数名称：parallel_kmp_worker
 * 函数功能：并行线程工作函数，负责局部匹配
 * 输入参数：text主串；pattern模式串；prefix前缀表；start/end区间范围；offset全局偏移；
 *           local_result输出匹配位置；thread_called记录线程调用
 ************************************************************/
void parallel_kmp_worker(const string& text, const string& pattern, const vector<int>& prefix, int start,
                         int end, int offset, vector<int>& local_result, vector<bool>& thread_called, int tid) {
    thread_called[tid] = true;
    string segment = text.substr(start, end-start);
    local_result = kmp_search(segment, pattern, prefix, offset);
}

/************************************************************
 * 函数名称：parallel_kmp
 * 函数功能：使用多个线程并行执行KMP匹配算法
 * 输入参数：text主串；pattern模式串；num_threads线程数；
 *           match_positions输出所有匹配位置；thread_called记录线程调用
 ************************************************************/
void parallel_kmp(const string& text, const string& pattern, int num_threads,
                  vector<int>& match_positions, vector<bool>& thread_called) {
    int n=text.size(), m=pattern.size();
    vector<int> prefix = build_kmp_table(pattern);
    vector<thread> threads;
    vector<vector<int>> thread_results(num_threads);
    thread_called.resize(num_threads, false);
    int chunk_size = n/num_threads;

    for (int i=0; i<num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads-1) ? n : (i+1)*chunk_size+m-1;
        if (end>n)
            end = n;
        threads.emplace_back(parallel_kmp_worker, cref(text), cref(pattern), cref(prefix), start, end, start,
                             ref(thread_results[i]), ref(thread_called), i);
    }

    for (auto& t : threads)
        t.join();

    match_positions.clear();
    for (const auto& vec:thread_results)
        match_positions.insert(match_positions.end(), vec.begin(), vec.end());
    sort(match_positions.begin(), match_positions.end());
}

/************************************************************
 * 函数名称：main
 * 函数功能：实验入口函数，生成随机数据并测试多线程匹配性能
 ************************************************************/
int main(int argc, char* argv[]) {
    if (argc<3) {
        cout << "Usage: " << argv[0] << " <pattern_length> <text_length>" << endl;
        return 1;
    }

    int pattern_len = stoi(argv[1]);
    int text_len = stoi(argv[2]);
    int insert_count = 5;

    string pattern = generate_random_string(pattern_len);
    string text = generate_random_string(text_len);
    vector<int> insert_positions;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, text_len - pattern_len - 1);
    for (int i = 0; i < insert_count; ++i)
        insert_positions.push_back(dis(gen));

    insert_pattern_at_positions(text, pattern, insert_positions);

    cout << "Fixed pattern: \"" << pattern << "\"" << endl;

    for (int num_threads : {1, 2, 4, 8}) {
        vector<int> matches;
        vector<bool> thread_called;

        auto start = chrono::high_resolution_clock::now();
        parallel_kmp(text, pattern, num_threads, matches, thread_called);
        auto end = chrono::high_resolution_clock::now();
        auto duration_ms = chrono::duration<double, milli>(end - start).count();

        cout << fixed << setprecision(2);
        cout << "Threads = " << num_threads << ", Time = " << duration_ms << " ms, Matches = " << matches.size();
        if (!matches.empty()) {
            cout << ", Positions: ";
            for (int i=0; i<min(5, (int)matches.size()); ++i)
                cout << matches[i] << " ";
            if (matches.size()>5)
                cout << "...";
        }
        cout << endl;

        for (int i =0; i<num_threads; ++i) {
            cout << "Thread " << i << ": " << (thread_called[i] ? "called" : "not called") << endl;
        }
        cout << endl;
    }

    return 0;
}


/*******************************************************
 * 文件名称：parallel_approx_match.cpp
 * 姓    名：陳日康
 * 学    号：22336049
 * 实验目的：实现并行近似字符串匹配算法（编辑距离不超过 k）
 * 程序说明：程序基于动态规划的带界编辑距离算法，并通过多线程。
 *          实现主串的大规模并行扫描。实验中支持设定模式串长度、
 *          主串长度和编辑距离容忍值 k，同时输出运行时间及线程调用
 *******************************************************/
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <chrono>
#include <random>
#include <iomanip>
using namespace std;

/*******************************************************
 * 函数名称：generate_random_string
 * 函数功能：生成由小写字母构成的随机字符串
 * 参    数：len 字符串长度
 * 返 回 值：生成的随机字符串
 *******************************************************/
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

/*******************************************************
 * 函数名称：insert_pattern_at_positions
 * 函数功能：将模式串插入到主串的指定位置
 * 参    数：text 主串，pattern 模式串，positions 插入位置数组
 *******************************************************/
void insert_pattern_at_positions(string& text, const string& pattern, const vector<int>& positions) {
    for (int pos:positions) {
        if (pos+pattern.size() <= text.size())
            copy(pattern.begin(), pattern.end(), text.begin() + pos);
    }
}

/*******************************************************
 * 函数名称：bounded_edit_distance
 * 函数功能：计算两个字符串之间的编辑距离（带界）
 * 参    数：text 比较的主串片段，pattern 模式串，k 最大允许编辑距离
 * 返 回 值：实际编辑距离（若超出 k 则返回 >k 值）
 *******************************************************/
int bounded_edit_distance(const string& text, const string& pattern, int k) {
    int n=text.size(), m=pattern.size();
    vector<vector<int>> dp(2, vector<int>(m + 1));
    for (int j=0; j<=m; ++j)
        dp[0][j] = j;

    for (int i=1; i<=n; ++i) {
        dp[i%2][0] = i;
        int min_val = dp[i%2][0];
        for (int j=1; j<=m; ++j) {
            if (text[i-1] == pattern[j-1])
                dp[i%2][j] = dp[(i-1) % 2][j-1];
            else
                dp[i%2][j] = 1 + min({ dp[(i-1) % 2][j], dp[i%2][j-1], dp[(i-1) % 2][j-1] });
            min_val = min(min_val, dp[i % 2][j]);
        }
        if (min_val>k)
            return k + 1;
    }
    return dp[n%2][m];
}

/*******************************************************
 * 函数名称：approximate_match_worker
 * 函数功能：线程工作函数，负责局部匹配并存储匹配位置
 * 参    数：text 主串，pattern 模式串，k 最大编辑距离，start/end 本线程处理的主串区间
 *           local_result 存储匹配结果的位置数组
 *******************************************************/
void approximate_match_worker(const string& text, const string& pattern, int k, int start, int end,
                              vector<int>& local_result, vector<bool>& thread_called, int tid) {
    thread_called[tid] = true;
    int m = pattern.size();
    for (int i=start; i<=end-m; ++i) {
        int dist = bounded_edit_distance(text.substr(i, m), pattern, k);
        if (dist<=k)
            local_result.push_back(i);
    }
}

/*******************************************************
 * 函数名称：parallel_approx_match
 * 函数功能：主函数，使用多线程并行执行近似字符串匹配
 * 参    数：text 主串。pattern 模式串。k 最大编辑距离
 *           num_threads 线程数，match_positions 输出的所有匹配位置，
 *           thread_results 每个线程单独的匹配结果
 *******************************************************/
void parallel_approx_match(const string& text, const string& pattern, int k, int num_threads, vector<int>& match_positions,
                            vector<vector<int>>& thread_results, vector<bool>& thread_called) {
    int n=text.size(), m=pattern.size();
    int chunk_size = n/num_threads;
    vector<thread> threads;
    thread_results.resize(num_threads);
    thread_called.resize(num_threads, false);

    for (int i=0; i<num_threads; ++i) {
        int start = i*chunk_size;
        int end = (i == num_threads-1) ? n : (i+1) * chunk_size+m-1;
        if (end>n)
            end = n;
        threads.emplace_back(approximate_match_worker, cref(text), cref(pattern), k, start, end, ref(thread_results[i]), ref(thread_called), i);
    }

    for (auto& t:threads)
        t.join();

    match_positions.clear();
    for (auto& vec:thread_results)
        match_positions.insert(match_positions.end(), vec.begin(), vec.end());
    sort(match_positions.begin(), match_positions.end());
}

/*******************************************************
 * 函数名称：main
 * 函数功能：主入口，初始化数据并执行多线程近似匹配测试
 *******************************************************/
int main(int argc, char* argv[]) {
    if (argc<4) {
        cout << "Usage: " << argv[0] << " <pattern_length> <text_length> <k>" << endl;
        return 1;
    }

    int pattern_len = stoi(argv[1]);
    int text_len = stoi(argv[2]);
    int k = stoi(argv[3]);
    int insert_count = 5;
    vector<int> insert_positions;

    string pattern = generate_random_string(pattern_len);
    string text = generate_random_string(text_len);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, text_len-pattern_len-1);
    for (int i=0; i<insert_count; ++i)
        insert_positions.push_back(dis(gen));

    insert_pattern_at_positions(text, pattern, insert_positions);

    cout << "Fixed pattern: \"" << pattern << "\"" << endl;

    for (int num_threads : {1, 2, 4, 8}) {
        vector<int> matches;
        vector<vector<int>> thread_results;
        vector<bool> thread_called;

        auto start = chrono::high_resolution_clock::now();
        parallel_approx_match(text, pattern, k, num_threads, matches, thread_results, thread_called);
        auto end = chrono::high_resolution_clock::now();
        double duration_ms = chrono::duration<double, milli>(end-start).count();

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

        for (int i=0; i<num_threads; ++i) {
            cout << "Thread " << i << ": " << (thread_called[i] ? "called" : "not called") << endl;
        }
        cout << endl;
    }

    return 0;
}


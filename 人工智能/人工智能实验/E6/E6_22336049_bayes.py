from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time

def read_data(file):
    with open(file, 'r', encoding='utf-8') as file_object:
        return [line.strip().split() for line in file_object]  # 按行读取并分割成单词列表

def laplace(res, k):
    row_sums = np.sum(res, axis=1) + k * res.shape[1]  # 计算每行的和并加上平滑因子
    return (res + k) / row_sums[:, None]  # 返回平滑后的结果

def extend(res):  # 扩大结果数值范围
    return res * 10000000000

def classify_emotions(set):  # 分类情感
    tv = TfidfVectorizer(use_idf=True)  # 创建TF-IDF向量化器
    tv_fit = tv.fit_transform(set)  # 将文本集合向量化
    ft_name = tv.get_feature_names_out()  # 获取特征名称
    res = tv_fit.toarray()  # 转换为数组形式
    res = laplace(res, 0.000114514)
    res_norm = extend(res)
    return res_norm, ft_name

def judge(num, arr, word_list, word):  # 判断情感类别
    res = 0
    for i in range(len(arr[num])):
        mul = 1
        for token in word_list:
            if token in word[num]:
                index = np.where(word[num] == token)[0][0]
                mul *= arr[num][i][index]  # 后验概率公式
        res += mul
    return res

def process_data(file_path):
    data_list = read_data(file_path)
    data_list.pop(0)
    emotions = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}  # 初始化情感字典
    for item in data_list:
        sentence = " ".join(item[3:])  # 合并句子部分
        emotions[int(item[1])].append(sentence)  # 按情感类别存储句子
    return emotions

def evaluate_model(test_data, arr, word):  # 评估模型
    right_ans = 0
    for i, test in enumerate(test_data):  # 遍历测试数据
        answer = int(test[1])  # 获得正确答案
        word_list = test[3:]  # 获取测试句子词列表
        max_score = float('-inf')  # 初始化最高分数
        judgement = -1  # 初始化判断结果
        for j in range(6):
            score = judge(j, arr, word_list, word)
            if score > max_score:  # 如果得分最高则更新
                max_score = score
                judgement = j + 1
        if judgement == answer:  # 如果判断正确
            right_ans += 1

        print("No.", i + 1)
        print("Judged:", judgement)
        print("Answer:", answer)
        print("Test case:", word_list)
        print()

    return (right_ans / len(test_data)) * 100

if __name__ == '__main__':
    start = time.perf_counter()
    training_data = process_data("train.txt")  # 处理训练数据
    arr = []
    word = []
    for emotion, sentences in training_data.items():  # 遍历训练数据的情感类别
        res_norm, ft_name = classify_emotions(sentences)  # 分类情感并取得结果
        arr.append(res_norm)
        word.append(ft_name)
    test_data = read_data("test.txt")  # 读取测试数据
    test_data.pop(0)
    accuracy = evaluate_model(test_data, arr, word)  # 评估模型

    print("Accuracy:", accuracy, "%")
    end = time.perf_counter()
    print("Total Time:", end - start, "s")
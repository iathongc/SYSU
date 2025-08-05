# -*- encoding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据读取函数
def load_data(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 初始化三个列表用于存储查询、上下文和答案
    queries, contexts, answers = [], [], []
    for line in lines:
        parts = line.strip().split('\t')
        # 检查行是否包含期望的4个部分
        if len(parts) == 4:
            queries.append(parts[1])
            contexts.append(parts[2])
            answers.append(parts[3])
    return queries, contexts, answers

# 构建词汇表
def build_vocab(sentences):
    vocab = {'<pad>': 0}  # 初始化词汇表，<pad>用于填充
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)  # 给每个新词分配一个新的索引
    return vocab

# 载入预训练词向量
def load_pretrained_embeddings(file_path, vocab, emb_dim):
    embeddings = np.zeros((len(vocab), emb_dim))  # 初始化嵌入矩阵
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            word = parts[0]
            if word in vocab:
                idx = vocab[word]
                try:
                    vector = np.array(parts[1:], dtype=float)
                except ValueError:
                    vector = np.random.uniform(-0.25, 0.25, emb_dim)  # 随机初始化无法解析的向量
                embeddings[idx] = vector
    return embeddings

# 自定义数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.vocab = vocab
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        # 将文本转换为索引序列
        self.processed_texts = [torch.tensor(self.text_to_indices(text), dtype=torch.long) for text in texts]

    def text_to_indices(self, text):
        # 将文本转换为词汇表中的索引
        return [self.vocab.get(word, self.vocab['<pad>']) for word in text.split()]

    def __len__(self):
        return len(self.labels)  # 返回数据集的大小

    def __getitem__(self, idx):
        return self.processed_texts[idx], self.labels[idx]

# 自定义collate函数
def pad_collate(batch):
    texts, labels = zip(*batch)
    # 对批次中的文本进行填充
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    return padded_texts, torch.tensor(labels)

# 定义LSTM分类器模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, num_classes, embeddings):
        super(LSTMClassifier, self).__init__()
        # 嵌入层，使用预训练词向量
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        # LSTM层，双向
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 全连接层，将LSTM的输出映射到分类结果
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # 嵌入层
        x, _ = self.lstm(x)  # LSTM层
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)  # 全连接层
        return x

# 训练与验证函数
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    scaler = GradScaler()  # 用于混合精度训练
    train_accuracies, train_losses, epoch_times = [], [], []

    for epoch in range(num_epochs):
        model.train()  # 模型设为训练模式
        start_time = time.time()
        epoch_loss, correct_predictions, total_samples = 0, 0, 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():  # 使用自动混合精度
                outputs = model(texts)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

        epoch_accuracy = correct_predictions / total_samples
        train_accuracies.append(epoch_accuracy)
        train_losses.append(epoch_loss / len(train_loader))
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.4f}, Time: {epoch_time:.2f}s')
        scheduler.step()  # 更新学习率

    # 验证阶段
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted_labels = torch.max(outputs, 1)
            val_correct += (predicted_labels == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Accuracy: {val_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # 绘制训练准确率和损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_accuracies, 'go-')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_losses, 'bo-')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()

    return train_accuracies, train_losses, epoch_times

# 主程序入口
if __name__ == "__main__":
    # 读取训练和验证数据
    train_queries, train_contexts, train_answers = load_data('train_40.tsv')
    val_queries, val_contexts, val_answers = load_data('dev_40.tsv')

    # 将查询和上下文合并成一个文本
    train_texts = [q + " " + c for q, c in zip(train_queries, train_contexts)]
    val_texts = [q + " " + c for q, c in zip(val_queries, val_contexts)]

    # 构建词汇表
    all_texts = train_texts + val_texts
    vocab = build_vocab(all_texts)

    # 载入预训练的词向量
    emb_dim = 300
    embedding_file = 'glove.840B.300d.txt'
    pretrained_embeddings = load_pretrained_embeddings(embedding_file, vocab, emb_dim)

    # 标签编码
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_answers)
    val_labels = label_encoder.transform(val_answers)

    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    val_dataset = TextDataset(val_texts, val_labels, vocab)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=pad_collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=pad_collate, num_workers=4)

    # 定义模型、损失函数、优化器和学习率调度器
    hidden_dim = 128
    num_layers = 2
    num_classes = len(np.unique(train_labels))
    model = LSTMClassifier(len(vocab), emb_dim, hidden_dim, num_layers, num_classes, pretrained_embeddings).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练和验证模型
    train_accuracies, train_losses, epoch_times = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10)

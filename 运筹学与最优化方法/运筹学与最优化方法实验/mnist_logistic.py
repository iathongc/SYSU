import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(__file__))

def load_mnist(batch_size=60000):       # 加载 MNIST 数据集
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 使用 DataLoader 加载数据
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader

# Logistic Regression 模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)     # 全连接层

    def forward(self, x):
        x = x.view(-1, 28 * 28)     # 将输入图层展平成一维向量
        return self.linear(x)

# 评估模型准确度的函数
@torch.no_grad()        # 在评估时不计算梯度
def evaluate(model, test_loader):
    model.eval()        # 设置模型为评估模式
    correct = total = 0
    for x, y in test_loader:
        out = model(x)      # 模型预测输出
        preds = out.argmax(dim=1)       # 获取预测的类别（最大值索引）
        correct += (preds == y).sum().item()    # 计算预测正确的数量
        total += y.size(0)      # 总样本数
    return correct / total

# 使用梯度下降法训练模型
def train_gd(epochs=10, lr=0.1):
    train_loader, test_loader = load_mnist(batch_size=60000)    # 加载数据
    model = LogisticRegression()          # 初始化模型
    criterion = nn.CrossEntropyLoss()     # 损失函数
    acc_list = []                         # 记录每轮的准确率

    for epoch in range(epochs):
        model.train()                   # 设置模型为训练模式
        for x, y in train_loader:           
            out = model(x)              # 前向传播
            loss = criterion(out, y)    # 计算损失

            model.zero_grad()           # 清空梯度
            loss.backward()             # 返向传播，计算梯度
            with torch.no_grad():       # 关闭梯度计算
                for param in model.parameters():
                    param -= lr * param.grad      # 更新权重

        acc = evaluate(model, test_loader)        # 计算测试集上的准确率
        acc_list.append(acc)            # 记录准确率
        print(f"[GD] Epoch {epoch+1}: Test Accuracy = {acc:.4f}")
    return acc_list

# 使用随机梯度下降法训练模型
def train_sgd(batch_size=64, epochs=10, lr=0.1):
    train_loader, test_loader = load_mnist(batch_size=batch_size)
    model = LogisticRegression()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)    # 使用 SGD 优化器
    acc_list = []

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()       # 清空梯度
            out = model(x)              # 向前传播
            loss = criterion(out, y)    # 计算损失
            loss.backward()             # 反向传播
            optimizer.step()

        acc = evaluate(model, test_loader)      # 计算测试集上的准确率
        acc_list.append(acc)
        print(f"[SGD bs={batch_size}] Epoch {epoch+1}: Test Accuracy = {acc:.4f}")
    return acc_list

if __name__ == '__main__':
    print()
    acc_gd = train_gd(epochs=10, lr=0.1)    
    print()
    acc_sgd_16 = train_sgd(batch_size=16, epochs=10, lr=0.1)
    print()
    acc_sgd_64 = train_sgd(batch_size=64, epochs=10, lr=0.1)
    print()
    acc_sgd_256 = train_sgd(batch_size=256, epochs=10, lr=0.1)
    print()

    plt.plot(acc_gd, label='GD (Full batch)')
    plt.plot(acc_sgd_16, label='SGD (batch=16)')
    plt.plot(acc_sgd_64, label='SGD (batch=64)')
    plt.plot(acc_sgd_256, label='SGD (batch=256)')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('MNIST Classification Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

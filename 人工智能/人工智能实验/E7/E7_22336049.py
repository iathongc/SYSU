import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

class NewCustomModel(nn.Module):
    def __init__(self):
        super(NewCustomModel, self).__init__()
        # 定义卷积部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),                      # 激活函数ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),      # 最大池化层，池化窗口2x2，步幅2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 定义全连接部分
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 14 * 14, 1024),   # 全连接层，将256*14*14的特征映射到1024个节点
            nn.ReLU(inplace=True),    # 激活函数ReLU
            nn.Dropout(0.5),          # Dropout层，丢弃50%的节点防止过拟合
            nn.Linear(1024, 512),   # 全连接层，将1024个节点映射到512个节点
            nn.ReLU(inplace=True),    # 激活函数ReLU
            nn.Dropout(0.5),          # Dropout层
            nn.Linear(512, 5)       # 全连接层，将512个节点映射到最终的5类
        )

    def forward(self, x):
        x = self.conv_layers(x)       # 通过卷积层
        x = x.view(x.size(0), -1)     # 展平操作
        x = self.fc_layers(x)         # 通过全连接层
        return x

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),    # 调整图像大小为224x224
    transforms.ToTensor(),            # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 标准化
])

# 加载数据集
train_data = DataLoader(
    ImageFolder(root='D:/python object/pythonProject/test/cnn_train', transform=data_transforms),
    batch_size=64, shuffle=True       # 训练集，批量大小64，打乱顺序
)
test_data = DataLoader(
    ImageFolder(root='D:/python object/pythonProject/test/cnn_test', transform=data_transforms),
    batch_size=64, shuffle=False      # 训练集，批量大小64，不打乱顺序
)

# 初始化模型、损失函数和优化器
net = NewCustomModel()                                # 实例化模型
loss_function = nn.CrossEntropyLoss()                 # 交叉熵损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)    # Adam优化器，学习率0.001

# 训练和评估过程
train_loss_values = []        # 记录每个epoch的训练损失
test_accuracy_values = []     # 记录每个epoch的测试准确率

for epoch in range(25):
    # 训练阶段
    net.train()
    total_loss = 0.0                              # 初始化总损失
    for images, labels in train_data:             # 遍历训练集
        optimizer.zero_grad()                     # 清空梯度
        outputs = net(images)                     # 前向传播
        loss = loss_function(outputs, labels)     # 计算损失
        loss.backward()                           # 反向传播
        optimizer.step()                          # 更新参数
        total_loss += loss.item()                 # 累计损失
    avg_loss = total_loss / len(train_data)       # 计算平均损失
    train_loss_values.append(avg_loss)            # 记录平均损失
    print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")      # 打印训练损失

    net.eval()                                    # 进入评估模式
    correct_predictions = 0                       # 初始化正确预测数
    total_samples = 0                             # 初始化总样本数
    with torch.no_grad():                         # 不计算梯度
        for images, labels in test_data:          # 遍历测试集
            outputs = net(images)                 # 前向传播
            _, preds = torch.max(outputs, 1)      # 获取预测结果
            total_samples += labels.size(0)       # 累计总样本数
            correct_predictions += (preds == labels).sum().item()       # 累计正确预测数
    accuracy = 100 * correct_predictions / total_samples                # 计算准确率
    test_accuracy_values.append(accuracy)                               # 记录准确率
    print(f"Test Accuracy: {accuracy:.2f}%")
print()
print("Training Complete.")        # 训练完成

# 绘制训练损失和测试准确率曲线
plt.figure(figsize=(12, 5))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, 26), train_loss_values, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()

# 绘制测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, 26), test_accuracy_values, label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

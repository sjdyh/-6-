#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

# Step 1: 数据加载与预处理
def load_images_with_labels(folder_path, image_size=(128, 128)):
    images = []
    labels = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            # 读取二进制文件
            with open(file_path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            # 检查文件大小是否匹配图像尺寸
            if data.size == image_size[0] * image_size[1]:
                image = data.reshape(image_size).T  # 转置以符合展示格式
                images.append(image)
                # 通过文件名规则提取标签
                # 假设文件名以 m_ 开头表示男性，以 f_ 开头表示女性
                if file_name.startswith("m_"):
                    labels.append(0)  # 男性
                elif file_name.startswith("f_"):
                    labels.append(1)  # 女性
                else:
                    print(f"无法识别文件名中的性别信息: {file_name}")
            else:
                print(f"文件 {file_name} 尺寸不匹配，跳过")
    return np.array(images), np.array(labels)

# 设置数据路径
folder_path = r"D:\face\rawdata"
images, labels = load_images_with_labels(folder_path)

# 数据预处理
X = images.reshape(len(images), -1) / 255.0  # 扁平化并归一化
y = labels

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 2: 设计 BP 神经网络
class BPNN(nn.Module):
    def __init__(self, input_size):
        super(BPNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一隐藏层
        self.fc2 = nn.Linear(128, 256)  # 第二隐藏层
        self.fc3 = nn.Linear(256, 1)  # 输出层
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

input_size = X_train.shape[1]
model = BPNN(input_size)

# Step 3: 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 4: 训练模型
epochs = 50
for epoch in tqdm(range(epochs), desc="Training"):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Step 5: 测试模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred >= 0.5).float()
    accuracy = (y_pred_class == y_test).sum().item() / y_test.size(0)
    print(f"Test Accuracy: {accuracy:.4f}")

# Step 6: 可视化示例图像
# 随机显示一个测试集图像和模型预测结果
plt.imshow(X_test[0].numpy().reshape(128, 128), cmap="gray")
plt.title(f"Predicted: {'Female' if y_pred_class[0].item() == 1 else 'Male'}")
plt.axis("off")
plt.show()


# In[ ]:





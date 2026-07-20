# 完整的代码实现，用于训练和验证一个简单的模型，输入为原始图片和预测图片的计数可信度
# R（连续类型）。代码包括数据加载、模型定义、训练、验证和测试过程，并在训练过程中保存最优模型。

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 定义数据集类
class VOCDataset(Dataset):
    def __init__(self, image_dir, detection_file, image_set_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = []
        self.labels = []

        # 读取图片名称和对应的可信度
        with open(image_set_file, 'r') as f:
            image_set = f.read().splitlines()

        with open(detection_file, 'r') as f:
            detection_results = f.read().splitlines()

        # 构建映射：图片名称 -> 可信度
        detection_map = {}
        for line in detection_results:
            parts = line.split(',')
            image_name = parts[0]  # 图片名称（带扩展名）
            confidence = float(parts[-1])
            detection_map[image_name] = confidence

        # 过滤出在image_set中的图片
        for image_name in image_set:
            # 为 image_name 添加 .jpg 后缀
            image_name_with_ext = image_name + '.jpg'
            if image_name_with_ext in detection_map:
                self.image_names.append(image_name_with_ext)
                self.labels.append(detection_map[image_name_with_ext])

        # 调试信息
        print(f"Loaded {len(self.image_names)} images from {image_set_file}")
        if len(self.image_names) == 0:
            print("Warning: No images loaded! Check file paths and contents.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        #print('image_path,label:',image_path,label)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # 输出一个连续值

    def forward(self, x):
        return self.backbone(x)


# 定义训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


# 定义验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


# 定义测试函数
def test(model, dataloader, criterion, device):
    # 加载最优模型
    best_model_path = os.path.join('model_realiablity/reliability', 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print(f"Error: Best model not found at {best_model_path}")
        return

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


# 绘制损失曲线
def plot_loss_curves(train_losses, val_losses, save_dir='loss_curves_realiablity/reliablity'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 每隔 10 个 epoch 保存一次图像
    if len(epochs) % 10 == 0:
        plt.savefig(os.path.join(save_dir, f'loss_curve_epoch_{len(epochs)}.png'))
    plt.close()


# 主函数
def main():
    # 超参数
    batch_size = 16
    num_epochs = 50  # 为了演示每隔 10 个 epoch 绘图
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    data_dir = './dataset/voc2069'
    train_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        detection_file=os.path.join(data_dir, 'detection_results.txt'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/train.txt'),
        transform=transform
    )
    val_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        detection_file=os.path.join(data_dir, 'detection_results.txt'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/val.txt'),
        transform=transform
    )
    test_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        detection_file=os.path.join(data_dir, 'detection_results.txt'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/test.txt'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()  # 回归任务使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建 models/ 文件夹（如果不存在）
    os.makedirs('model_realiablity', exist_ok=True)

    # 训练和验证
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join('model_realiablity/reliability', 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

        # 每 10 个 epoch 保存模型和绘制损失曲线
        if (epoch + 1) % 10 == 0:
            plot_loss_curves(train_losses, val_losses)

            model_save_path = os.path.join('model_realiablity/reliability', f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    # 测试
    test_loss = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')


if __name__ == '__main__':
    main()

#Mean Absolute Error (MAE) on test set: 0.0618


"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 定义数据集类
class VOCDataset(Dataset):
    def __init__(self, image_dir, detection_file, image_set_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = []
        self.labels = []

        # 读取图片名称和对应的可信度
        with open(image_set_file, 'r') as f:
            image_set = f.read().splitlines()

        with open(detection_file, 'r') as f:
            detection_results = f.read().splitlines()

        # 构建映射：图片名称 -> 可信度
        detection_map = {}
        for line in detection_results:
            parts = line.split(',')
            image_name = parts[0]  # 图片名称（带扩展名）
            confidence = float(parts[-1])
            detection_map[image_name] = confidence

        # 过滤出在image_set中的图片
        for image_name in image_set:
            # 为 image_name 添加 .jpg 后缀
            image_name_with_ext = image_name + '.jpg'
            if image_name_with_ext in detection_map:
                self.image_names.append(image_name_with_ext)
                self.labels.append(detection_map[image_name_with_ext])

        # 调试信息
        print(f"Loaded {len(self.image_names)} images from {image_set_file}")
        if len(self.image_names) == 0:
            print("Warning: No images loaded! Check file paths and contents.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        #print('image, label:', image_path, label)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # 输出一个连续值

    def forward(self, x):
        return self.backbone(x)


# 定义损失函数和优化器
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)


        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


# 绘制损失曲线
def plot_loss_curves(train_losses, val_losses, save_dir='loss_curves'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 每隔 10 个 epoch 保存一次图像
    if len(epochs) % 10 == 0:
        plt.savefig(os.path.join(save_dir, f'loss_curve_epoch_{len(epochs)}.png'))
    plt.close()


# 主函数
def main():
    # 超参数
    batch_size = 16
    num_epochs = 100  # 为了演示每隔 10 个 epoch 绘图
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    data_dir = './dataset/voc2069'
    train_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        detection_file=os.path.join(data_dir, 'detection_results.txt'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/train.txt'),
        transform=transform
    )
    val_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        detection_file=os.path.join(data_dir, 'detection_results.txt'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/val.txt'),
        transform=transform
    )
    test_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        detection_file=os.path.join(data_dir, 'detection_results.txt'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/test.txt'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()  # 回归任务使用均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        # 每隔 10 个 epoch 绘制损失曲线
        if (epoch + 1) % 10 == 0:
            plot_loss_curves(train_losses, val_losses)
            # 保存模型
            model_save_path = f'model_epoch_{epoch + 1}.pth'  # 使用 epoch+1 作为文件名
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")


    # 测试
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')


if __name__ == '__main__':
    main()
"""

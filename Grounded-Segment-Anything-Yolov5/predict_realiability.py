import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# 定义数据集类
class VOCDataset:
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

        # 返回原始图像和预处理后的图像
        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        return image, transformed_image, torch.tensor(label, dtype=torch.float32), image_name


# 自定义 collate_fn
def custom_collate_fn(batch):
    original_images = [item[0] for item in batch]  # PIL.Image.Image
    transformed_images = torch.stack([item[1] for item in batch])  # torch.Tensor
    labels = torch.stack([item[2] for item in batch])  # torch.Tensor
    image_names = [item[3] for item in batch]  # str
    return original_images, transformed_images, labels, image_names


# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 使用最新的权重加载方式
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # 输出一个连续值

    def forward(self, x):
        return self.backbone(x)


# 定义推理函数
def predict(model, dataloader, device, save_dir='results_realiability/reliability'):
    # 创建保存结果的文件夹
    os.makedirs(save_dir, exist_ok=True)

    # 设置模型为评估模式
    model.eval()

    # 初始化绝对误差列表
    absolute_errors = []

    # 遍历测试集
    for original_images, transformed_images, labels, image_names in dataloader:
        transformed_images = transformed_images.to(device)
        labels = labels.to(device)

        # 推理
        with torch.no_grad():
            outputs = model(transformed_images).squeeze()

        # 将结果保存到原始图像上
        for i in range(len(original_images)):
            # 获取原始图像
            original_image = original_images[i]
            draw = ImageDraw.Draw(original_image)

            # 设置更大的字体
            try:
                # 尝试加载指定字体文件（确保路径正确）
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 替换为实际字体路径
                font = ImageFont.truetype(font_path, 80)  # 字体大小设置为 40
            except IOError:
                # 如果找不到字体文件，使用默认字体并调整大小
                font = ImageFont.load_default()
                print("Warning: Specified font not found. Using default font.")

            # 获取预测值和 ground truth
            if outputs.dim() == 0:  # 如果 outputs 是标量
                pred_realiability = outputs.item()
            else:  # 如果 outputs 是 1 维张量
                pred_realiability = outputs[i].item()
            gt_realiability = labels[i].item()

            # 计算绝对误差
            absolute_error = abs(pred_realiability - gt_realiability)
            absolute_errors.append(absolute_error)

            # 在原始图像上绘制文本
            draw.text((200, 100), f"pred_realiability: {pred_realiability:.4f}", fill="blue", font=font)
            draw.text((200, 200), f"gt_realiability: {gt_realiability:.4f}", fill="green", font=font)
            draw.text((200, 300), f"absolute_error: {absolute_error:.4f}", fill="red", font=font)

            # 保存结果图像
            result_path = os.path.join(save_dir, image_names[i])
            original_image.save(result_path)
            print(f"Saved result to {result_path}")

    # 计算平均绝对误差
    mae = np.mean(absolute_errors)
    print(f"Mean Absolute Error (MAE) on test set: {mae:.4f}")


# 主函数
def main():
    # 超参数
    batch_size = 1  # 逐张图像推理
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    data_dir = './dataset/voc2069'
    test_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        detection_file=os.path.join(data_dir, 'detection_results.txt'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/test.txt'),
        transform=transform
    )

    # 使用自定义的 collate_fn
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 初始化模型
    model = SimpleModel().to(device)

    # 加载最佳模型
    best_model_path = os.path.join('model_realiablity/reliability', 'best_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print(f"Error: Best model not found at {best_model_path}")
        return

    # 推理并保存结果
    predict(model, test_loader, device)


if __name__ == '__main__':
    main()


# Mean Absolute Error (MAE) on test set: 0.0600
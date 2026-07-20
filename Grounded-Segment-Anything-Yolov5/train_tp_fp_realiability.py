# 完整的代码实现
'''
多任务模型：
模型同时输出分类结果（TP 和 FP）和回归结果（reliability）。
使用 ResNet18 作为主干网络，并在最后添加两个全连接层分别用于分类和回归。
损失函数：
分类任务使用交叉熵损失（nn.CrossEntropyLoss）。
回归任务使用均方误差损失（nn.MSELoss）。
训练和验证：
在训练和验证阶段，同时计算分类损失和回归损失。
保存最优模型和每 10 个 epoch 的模型。
测试：在测试阶段，计算分类损失和回归损失。
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 定义数据集类
class VOCDataset(Dataset):
    def __init__(self, image_dir, predict_dir, label_dir, image_set_file, transform=None, iou_threshold=0.5):
        self.image_dir = image_dir
        self.predict_dir = predict_dir
        self.label_dir = label_dir
        self.transform = transform
        self.iou_threshold = iou_threshold
        self.image_names = []

        # 读取图片名称
        with open(image_set_file, 'r') as f:
            image_set = f.read().splitlines()

        # 过滤出在image_set中的图片
        for image_name in image_set:
            self.image_names.append(image_name + '.jpg')

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
        #print('image_path:',image_path)

        # 读取预测结果和 ground truth
        predict_path = os.path.join(self.predict_dir, image_name.replace('.jpg', '.txt'))
        label_path = os.path.join(self.label_dir, image_name.replace('.jpg', '.txt'))

        # 解析预测结果和 ground truth
        predict_boxes = self.parse_yolo_file(predict_path)
        gt_boxes = self.parse_yolo_file(label_path)
        #print('predict_path,label_path:',predict_path,label_path)
        #print('predict_boxes:',predict_boxes)
        #print('gt_boxes:', gt_boxes)

        # 计算 TP、FP 和 FN
        tp_boxes, fp_boxes, fn_boxes = self.calculate_tp_fp_fn(predict_boxes, gt_boxes)

        # 计算 reliability
        reliability = len(tp_boxes) / (len(tp_boxes) + len(fp_boxes) + len(fn_boxes)) if (len(tp_boxes) + len(fp_boxes) + len(fn_boxes)) > 0 else 0.0

        # 从原始图像中裁剪出 TP 和 FP 的区域
        tp_images = [self.crop_image(image, box) for box in tp_boxes]
        fp_images = [self.crop_image(image, box) for box in fp_boxes]

        # 合并 TP 和 FP 的图像和标签
        images = tp_images + fp_images
        labels = [0] * len(tp_boxes) + [1] * len(fp_boxes)

        # 如果没有 TP 或 FP，添加一个占位符
        if len(images) == 0:
            placeholder_image = Image.new('RGB', (224, 224), (0, 0, 0))  # 全零图像
            images.append(placeholder_image)
            labels.append(-1)  # 使用一个无效标签

        # 转换为 Tensor
        if self.transform:
            images = [self.transform(img) for img in images]
            image = self.transform(image)

        return images, torch.tensor(labels, dtype=torch.long), image, torch.tensor(reliability, dtype=torch.float32)

    def parse_yolo_file(self, file_path):
        boxes = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5 or len(parts) == 6:
                        cls = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        boxes.append((cls, x_center, y_center, width, height))
        return boxes

    def calculate_tp_fp_fn(self, predict_boxes, gt_boxes, iou_threshold=0.5):
        tp_boxes = []  # True Positives
        fp_boxes = []  # False Positives
        fn_boxes = gt_boxes.copy()  # False Negatives（初始为所有真实框）

        # 用于记录已经匹配的真实框
        matched_gt_indices = set()

        for p_box in predict_boxes:
            is_tp = False
            best_iou = 0.0
            best_gt_index = -1

            # 遍历所有真实框，找到与当前预测框 IoU 最大的真实框
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt_indices:
                    continue  # 跳过已经匹配的真实框
                iou = self.calculate_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = i

            # 如果最大 IoU 大于阈值，则认为是 TP
            if best_iou >= iou_threshold:
                tp_boxes.append(p_box)
                matched_gt_indices.add(best_gt_index)  # 标记该真实框已匹配
            else:
                fp_boxes.append(p_box)  # 否则是 FP

        # 计算 FN
        fn_boxes = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt_indices]

        # 计算混淆矩阵
        tp = len(tp_boxes)
        fp = len(fp_boxes)
        fn = len(fn_boxes)

        #print("Confusion Matrix:")
        #print(f"TP: {tp}, FP: {fp}, FN: {fn}")

        return tp_boxes, fp_boxes, fn_boxes

    def calculate_iou(self, box1, box2):
        # 提取框的坐标
        x1_center, y1_center, w1, h1 = box1[1], box1[2], box1[3], box1[4]
        x2_center, y2_center, w2, h2 = box2[1], box2[2], box2[3], box2[4]

        # 将中心坐标转换为边界坐标
        x1_min = x1_center - w1 / 2
        y1_min = y1_center - h1 / 2
        x1_max = x1_center + w1 / 2
        y1_max = y1_center + h1 / 2

        x2_min = x2_center - w2 / 2
        y2_min = y2_center - h2 / 2
        x2_max = x2_center + w2 / 2
        y2_max = y2_center + h2 / 2

        # 计算交集区域的坐标
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # 计算交集区域的面积
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # 计算并集区域的面积
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        # 计算 IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def crop_image(self, image, box):
        # 从原始图像中裁剪出 bounding box 对应的区域
        img_width, img_height = image.size
        x_center, y_center, width, height = box[1], box[2], box[3], box[4]
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image = cropped_image.resize((224, 224))  # 调整裁剪图像的大小
        return cropped_image

# 自定义 collate_fn
def custom_collate_fn(batch):
    images_list = [item[0] for item in batch]  # 每个样本的 images 列表
    labels_list = [item[1].tolist() for item in batch]  # 将 labels 转换为列表
    original_images = torch.stack([item[2] for item in batch])  # 原始图像
    reliability = torch.stack([item[3] for item in batch])  # reliability

    # 找到最长的 images 列表
    max_len = max(len(images) for images in images_list)

    # 填充 images 和 labels
    padded_images_list = []
    padded_labels_list = []
    for images, labels in zip(images_list, labels_list):
        if len(images) < max_len:
            # 填充占位符图像和无效标签
            placeholder_image = torch.zeros_like(images[0])  # 全零图像
            images.extend([placeholder_image] * (max_len - len(images)))
            labels.extend([-1] * (max_len - len(labels)))  # 使用列表的 extend 方法
        padded_images_list.append(torch.stack(images))
        padded_labels_list.append(torch.tensor(labels))  # 将填充后的 labels 转换为 Tensor

    # 将填充后的 images 和 labels 转换为张量
    padded_images = torch.stack(padded_images_list)
    padded_labels = torch.stack(padded_labels_list)

    return padded_images, padded_labels, original_images, reliability


# 定义多任务模型
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 512)
        self.classifier = nn.Linear(512, 2)  # 分类任务：TP 和 FP
        self.regressor = nn.Linear(512, 1)   # 回归任务：reliability

    def forward(self, images, original_images):
        # 处理裁剪后的图像
        batch_size, num_images_per_sample, channels, height, width = images.shape
        images = images.view(-1, channels, height, width)  # [batch_size * num_images_per_sample, channels, height, width]
        features = self.backbone(images)  # [batch_size * num_images_per_sample, 512]

        # 处理原始图像
        original_features = self.backbone(original_images)  # [batch_size, 512]

        # 将原始图像的特征与裁剪图像的特征结合
        original_features = original_features.unsqueeze(1).repeat(1, num_images_per_sample, 1)  # [batch_size, num_images_per_sample, 512]
        original_features = original_features.view(-1, 512)  # [batch_size * num_images_per_sample, 512]

        # 结合特征
        combined_features = features + original_features  # [batch_size * num_images_per_sample, 512]

        # 分类和回归输出
        classification_output = self.classifier(combined_features)  # [batch_size * num_images_per_sample, 2]
        regression_output = self.regressor(combined_features)  # [batch_size * num_images_per_sample, 1]

        return classification_output, regression_output

# 定义训练函数
def train(model, dataloader, criterion_cls, criterion_reg, optimizer, device):
    model.train()
    running_loss_cls = 0.0
    running_loss_reg = 0.0
    for images, labels, original_images, reliability in dataloader:
        # 调整 labels 和 reliability 的形状
        batch_size, num_images_per_sample = images.shape[:2]
        labels = labels.view(-1)  # [batch_size * num_images_per_sample]
        reliability = reliability.unsqueeze(1).repeat(1, num_images_per_sample).view(-1)  # [batch_size * num_images_per_sample]

        images = images.to(device)
        labels = labels.to(device)
        original_images = original_images.to(device)
        reliability = reliability.to(device)

        optimizer.zero_grad()
        classification_output, regression_output = model(images, original_images)  # 传入裁剪图像和原始图像
        loss_cls = criterion_cls(classification_output, labels)  # 分类损失
        loss_reg = criterion_reg(regression_output.squeeze(), reliability)  # 回归损失
        loss = loss_cls + loss_reg
        loss.backward()
        optimizer.step()

        running_loss_cls += loss_cls.item() * batch_size
        running_loss_reg += loss_reg.item() * batch_size
    return running_loss_cls / len(dataloader.dataset), running_loss_reg / len(dataloader.dataset)

# 定义验证函数
def validate(model, dataloader, criterion_cls, criterion_reg, device):
    model.eval()
    running_loss_cls = 0.0
    running_loss_reg = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels, original_images, reliability in dataloader:
            # 调整 labels 和 reliability 的形状
            batch_size, num_images_per_sample = images.shape[:2]
            labels = labels.view(-1)  # [batch_size * num_images_per_sample]
            reliability = reliability.unsqueeze(1).repeat(1, num_images_per_sample).view(-1)  # [batch_size * num_images_per_sample]

            images = images.to(device)
            labels = labels.to(device)
            original_images = original_images.to(device)
            reliability = reliability.to(device)

            classification_output, regression_output = model(images, original_images)  # 传入裁剪图像和原始图像
            loss_cls = criterion_cls(classification_output, labels)
            loss_reg = criterion_reg(regression_output.squeeze(), reliability)
            running_loss_cls += loss_cls.item() * batch_size
            running_loss_reg += loss_reg.item() * batch_size

            # 收集分类结果
            _, preds = torch.max(classification_output, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算分类指标
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    mAP = average_precision_score(all_labels, all_preds)

    return running_loss_cls / len(dataloader.dataset), running_loss_reg / len(dataloader.dataset), accuracy, mAP

# 定义测试函数
def test(model, dataloader, criterion_cls, criterion_reg, device):
    # 加载最优模型
    best_model_path = os.path.join('model_realiablity/tp_fp_reliability', 'best_model.pth')
    #best_model_path = os.path.join('model_realiablity/tp_fp_reliability', 'model_epoch_10.pth')

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print(f"Error: Best model not found at {best_model_path}")
        return

    model.eval()
    running_loss_cls = 0.0
    running_loss_reg = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels, original_images, reliability in dataloader:
            # 调整 labels 和 reliability 的形状
            batch_size, num_images_per_sample = images.shape[:2]
            labels = labels.view(-1)  # [batch_size * num_images_per_sample]
            reliability = reliability.unsqueeze(1).repeat(1, num_images_per_sample).view(-1)  # [batch_size * num_images_per_sample]

            images = images.to(device)
            labels = labels.to(device)
            original_images = original_images.to(device)
            reliability = reliability.to(device)

            classification_output, regression_output = model(images, original_images)  # 传入裁剪图像和原始图像
            loss_cls = criterion_cls(classification_output, labels)
            loss_reg = criterion_reg(regression_output.squeeze(), reliability)
            running_loss_cls += loss_cls.item() * batch_size
            running_loss_reg += loss_reg.item() * batch_size

            # 收集分类结果
            _, preds = torch.max(classification_output, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算分类指标
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    mAP = average_precision_score(all_labels, all_preds)

    return running_loss_cls / len(dataloader.dataset), running_loss_reg / len(dataloader.dataset), accuracy, mAP



# 绘制损失曲线
def plot_loss_curves(train_losses_cls, val_losses_cls, train_losses_reg, val_losses_reg, save_dir='loss_curves_realiablity/tp_fp_reliability'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses_cls) + 1)

    plt.figure()
    plt.plot(epochs, train_losses_cls, label='Train Classification Loss')
    plt.plot(epochs, val_losses_cls, label='Validation Classification Loss')
    plt.plot(epochs, train_losses_reg, label='Train Regression Loss')
    plt.plot(epochs, val_losses_reg, label='Validation Regression Loss')
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
    batch_size = 1
    num_epochs = 50
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    data_dir = './dataset/voc2069'
    train_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        predict_dir=os.path.join(data_dir, 'predict_labels'),
        label_dir=os.path.join(data_dir, 'labels'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/train.txt'),
        transform=transform
    )
    val_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        predict_dir=os.path.join(data_dir, 'predict_labels'),
        label_dir=os.path.join(data_dir, 'labels'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/val.txt'),
        transform=transform
    )
    test_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        predict_dir=os.path.join(data_dir, 'predict_labels'),
        label_dir=os.path.join(data_dir, 'labels'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/test.txt'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 初始化模型、损失函数和优化器
    model = MultiTaskModel().to(device)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)  # 忽略无效标签
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建 models/ 文件夹（如果不存在）
    os.makedirs('model_realiablity', exist_ok=True)

    # 训练和验证
    train_losses_cls = []
    val_losses_cls = []
    train_losses_reg = []
    val_losses_reg = []
    val_accuracies = []
    val_mAPs = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss_cls, train_loss_reg = train(model, train_loader, criterion_cls, criterion_reg, optimizer, device)
        train_losses_cls.append(train_loss_cls)
        train_losses_reg.append(train_loss_reg)

        val_loss_cls, val_loss_reg, val_accuracy, val_mAP = validate(model, val_loader, criterion_cls, criterion_reg, device)
        val_losses_cls.append(val_loss_cls)
        val_losses_reg.append(val_loss_reg)
        val_accuracies.append(val_accuracy)
        val_mAPs.append(val_mAP)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Classification Loss: {train_loss_cls:.4f}, Train Regression Loss: {train_loss_reg:.4f}, '
              f'Val Classification Loss: {val_loss_cls:.4f}, Val Regression Loss: {val_loss_reg:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}, Val mAP: {val_mAP:.4f}')

        # 保存最优模型
        if val_loss_cls + val_loss_reg < best_val_loss:
            best_val_loss = val_loss_cls + val_loss_reg
            best_model_path = os.path.join('model_realiablity/tp_fp_reliability', 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")

        # 每 10 个 epoch 保存模型和绘制损失曲线
        if (epoch + 1) % 10 == 0:
            plot_loss_curves(train_losses_cls, val_losses_cls, train_losses_reg, val_losses_reg)

            model_save_path = os.path.join('model_realiablity/tp_fp_reliability', f'model_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

    # 测试
    test_loss_cls, test_loss_reg, test_accuracy, test_mAP = test(model, test_loader, criterion_cls, criterion_reg, device)
    print(f'Test Classification Loss: {test_loss_cls:.4f}, Test Regression Loss: {test_loss_reg:.4f}, '
          f'Test Accuracy: {test_accuracy:.4f}, Test mAP: {test_mAP:.4f}')

if __name__ == '__main__':
    main()




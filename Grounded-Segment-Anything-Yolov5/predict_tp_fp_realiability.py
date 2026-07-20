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
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# 使用默认字体
font = ImageFont.load_default()

# 定义数据集类
class VOCDataset:
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

        # 读取预测结果和 ground truth
        predict_path = os.path.join(self.predict_dir, image_name.replace('.jpg', '.txt'))
        label_path = os.path.join(self.label_dir, image_name.replace('.jpg', '.txt'))

        # 解析预测结果和 ground truth
        predict_boxes = self.parse_yolo_file(predict_path)
        gt_boxes = self.parse_yolo_file(label_path)

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

        # 打印调试信息
        print(f"Number of images: {len(images)}")  # 应该是 TP + FP 的数量
        print(f"Labels: {labels}")  # 应该是 TP 和 FP 的标签

        return images, torch.tensor(labels, dtype=torch.long), image, torch.tensor(reliability, dtype=torch.float32), image_name

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
    image_names = [item[4] for item in batch]  # 图像名称

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
        padded_images_list.append(torch.stack(images))  # 将 images 堆叠为 [num_images_per_sample, channels, height, width]
        padded_labels_list.append(torch.tensor(labels))  # 将填充后的 labels 转换为 Tensor

    # 将填充后的 images 和 labels 转换为张量
    padded_images = torch.stack(padded_images_list)  # [batch_size, num_images_per_sample, channels, height, width]
    padded_labels = torch.stack(padded_labels_list)  # [batch_size, num_images_per_sample]

    return padded_images, padded_labels, original_images, reliability, image_names

# 定义多任务模型
class MultiTaskModel(torch.nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)  # 使用 torchvision 的 ResNet18
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, 512)
        self.classifier = torch.nn.Linear(512, 2)  # 分类任务：TP 和 FP
        self.regressor = torch.nn.Linear(512, 1)   # 回归任务：reliability

    def forward(self, images, original_images):
        # 处理裁剪后的图像
        batch_size_times_num_images, channels, height, width = images.shape
        batch_size = original_images.shape[0]
        num_images_per_sample = batch_size_times_num_images // batch_size

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

# 绘制框和文本
def draw_boxes_and_text__(image, tp_boxes, fp_boxes, pred_reliability, gt_reliability, absolute_error):
    draw = ImageDraw.Draw(image)

    # 设置字体大小
    font_size = 20  # 可调整字体大小
    try:
        # 尝试加载指定字体
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 替换为实际的路径
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        # 如果字体文件不存在，使用默认字体（不支持大小调整）
        print("Warning: Specified font not found. Using default font.")
        font = ImageFont.load_default()
        font_1 = ImageFont.truetype(font_path, size=80)

    # 绘制 TP 框（绿色）
    for box in tp_boxes:
        cls, x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image.width)
        y1 = int((y_center - height / 2) * image.height)
        x2 = int((x_center + width / 2) * image.width)
        y2 = int((y_center + height / 2) * image.height)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=5)  # 绿色
        # 在矩形框左上角绘制文本
        text = f"TP: {cls}"
        draw.text((x1 + 5, y1 + 5), text, fill=(0, 255, 0), font=font)  # 绿色，偏移 5 像素

    # 绘制 FP 框（蓝色）
    for box in fp_boxes:
        cls, x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image.width)
        y1 = int((y_center - height / 2) * image.height)
        x2 = int((x_center + width / 2) * image.width)
        y2 = int((y_center + height / 2) * image.height)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=5)  # 蓝色
        # 在矩形框左上角绘制文本
        text = f"FP: {cls}"
        draw.text((x1 + 5, y1 + 5), text, fill=(0, 0, 255), font=font)  # 蓝色，偏移 5 像素

    # 绘制文本（预测可靠性、真实可靠性、绝对误差）
    draw.text((10, 10), f"Pred Reliability: {pred_reliability:.4f}", fill=(255, 0, 0), font=font)  # 红色
    draw.text((10, 30), f"GT Reliability: {gt_reliability:.4f}", fill=(255, 0, 0), font=font)  # 红色
    draw.text((10, 50), f"Absolute Error: {absolute_error:.4f}", fill=(255, 0, 0), font=font)  # 红色

    return image

def draw_boxes_and_text(image, tp_boxes, fp_boxes, pred_reliability, gt_reliability, absolute_error):
    draw = ImageDraw.Draw(image)

    # 设置字体大小
    font_size = 20  # 可调整字体大小
    try:
        # 尝试加载指定字体
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # 替换为实际的路径
        font = ImageFont.truetype(font_path, size=font_size)
        font_1 = ImageFont.truetype(font_path, size=80)
    except IOError:
        # 如果字体文件不存在，使用默认字体（不支持大小调整）
        print("Warning: Specified font not found. Using default font.")
        font = ImageFont.load_default()

    # 绘制 TP 框（绿色）
    for box in tp_boxes:
        cls, x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image.width)
        y1 = int((y_center - height / 2) * image.height)
        x2 = int((x_center + width / 2) * image.width)
        y2 = int((y_center + height / 2) * image.height)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=5)  # 绿色
        # 在矩形框左上角绘制文本
        text = f"TP: {cls}"
        draw.text((x1 + 5, y1 + 5), text, fill=(0, 255, 0), font=font)  # 绿色，偏移 5 像素

    # 绘制 FP 框（蓝色）
    for box in fp_boxes:
        cls, x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image.width)
        y1 = int((y_center - height / 2) * image.height)
        x2 = int((x_center + width / 2) * image.width)
        y2 = int((y_center + height / 2) * image.height)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 255), width=5)  # 蓝色
        # 在矩形框左上角绘制文本
        text = f"FP: {cls}"
        draw.text((x1 + 5, y1 + 5), text, fill=(0, 0, 255), font=font)  # 蓝色，偏移 5 像素


    # 绘制文本（预测可靠性、真实可靠性、绝对误差）
    draw.text((200, 100), f"Pred Reliability: {pred_reliability:.4f}", fill="blue", font=font_1)
    draw.text((200, 200), f"GT Reliability: {gt_reliability:.4f}", fill="green", font=font_1)
    draw.text((200, 300), f"Absolute Error: {absolute_error:.4f}", fill="red", font=font_1)

    return image


# 推理函数
def predict(model, test_loader, device, save_dir='./results_realiability/tp_fp_reliability'):
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (images, labels, original_images, reliability, image_names) in enumerate(test_loader):
        # 调整输入形状
        batch_size, num_images_per_sample, channels, height, width = images.shape
        images = images.view(-1, channels, height, width).to(device)  # [batch_size * num_images_per_sample, channels, height, width]
        original_images = original_images.to(device)

        # 推理
        with torch.no_grad():
            classification_output, regression_output = model(images, original_images)

        # 获取预测结果
        _, preds = torch.max(classification_output, 1)
        pred_reliability = regression_output.mean().item()  # 预测的可靠性
        gt_reliability = reliability.mean().item()  # 真实的可靠性
        absolute_error = abs(pred_reliability - gt_reliability)  # 绝对误差
        print('preds:', preds)

        # 获取 TP 和 FP 框
        tp_boxes = []
        fp_boxes = []
        image_name = image_names[0]  # 当前图像名称
        predict_file_path = os.path.join(test_loader.dataset.predict_dir, image_name.replace('.jpg', '.txt'))
        predict_boxes = test_loader.dataset.parse_yolo_file(predict_file_path)  # 解析预测框

        # 将 preds 与 predict_boxes 对应起来
        for i, pred in enumerate(preds):
            if i < len(predict_boxes):  # 确保 preds 和 predict_boxes 的长度一致
                if pred == 0:  # TP
                    tp_boxes.append(predict_boxes[i])
                else:  # FP
                    fp_boxes.append(predict_boxes[i])

        # 加载原始图像
        original_image_path = os.path.join(test_loader.dataset.image_dir, image_names[0])
        original_image = Image.open(original_image_path).convert('RGB')

        # 在原图上绘制 TP 和 FP 框以及文本
        result_image = draw_boxes_and_text(original_image, tp_boxes, fp_boxes, pred_reliability, gt_reliability, absolute_error)

        # 保存结果图像
        result_image.save(os.path.join(save_dir, image_names[0]))
        print(f"Saved result for image: {image_names[0]}")

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试集
    data_dir = './dataset/voc2069'
    test_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        predict_dir=os.path.join(data_dir, 'predict_labels'),
        label_dir=os.path.join(data_dir, 'labels'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/test.txt'),
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # 加载最佳模型
    best_model_path = './model_realiablity/tp_fp_reliability/best_model.pth'
    model = MultiTaskModel().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 进行推理
    predict(model, test_loader, device)

if __name__ == '__main__':
    main()




"""
import os
import torch
import torchvision  # 导入 torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 定义数据集类
class VOCDataset:
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

        # 读取预测结果和 ground truth
        predict_path = os.path.join(self.predict_dir, image_name.replace('.jpg', '.txt'))
        label_path = os.path.join(self.label_dir, image_name.replace('.jpg', '.txt'))

        # 解析预测结果和 ground truth
        predict_boxes = self.parse_yolo_file(predict_path)
        gt_boxes = self.parse_yolo_file(label_path)

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

        # 打印调试信息
        print(f"Number of images: {len(images)}")  # 应该是 TP + FP 的数量
        print(f"Labels: {labels}")  # 应该是 TP 和 FP 的标签

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
        padded_images_list.append(torch.stack(images))  # 将 images 堆叠为 [num_images_per_sample, channels, height, width]
        padded_labels_list.append(torch.tensor(labels))  # 将填充后的 labels 转换为 Tensor

    # 将填充后的 images 和 labels 转换为张量
    padded_images = torch.stack(padded_images_list)  # [batch_size, num_images_per_sample, channels, height, width]
    padded_labels = torch.stack(padded_labels_list)  # [batch_size, num_images_per_sample]

    # 打印调试信息
    print(f"padded_images shape: {padded_images.shape}")  # 应该是 [batch_size, num_images_per_sample, channels, height, width]
    print(f"padded_labels shape: {padded_labels.shape}")  # 应该是 [batch_size, num_images_per_sample]

    return padded_images, padded_labels, original_images, reliability

# 定义多任务模型
class MultiTaskModel(torch.nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)  # 使用 torchvision 的 ResNet18
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, 512)
        self.classifier = torch.nn.Linear(512, 2)  # 分类任务：TP 和 FP
        self.regressor = torch.nn.Linear(512, 1)   # 回归任务：reliability

    def forward(self, images, original_images):
        # 打印调试信息
        print(f"images shape in forward: {images.shape}")  # 应该是 [batch_size * num_images_per_sample, channels, height, width]

        # 处理裁剪后的图像
        batch_size_times_num_images, channels, height, width = images.shape
        batch_size = original_images.shape[0]
        num_images_per_sample = batch_size_times_num_images // batch_size

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

# 绘制框和文本
def draw_boxes_and_text(image, tp_boxes, fp_boxes, pred_reliability, gt_reliability, absolute_error):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # 绘制 TP 框（绿色）
    for box in tp_boxes:
        cls, x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image.width)
        y1 = int((y_center - height / 2) * image.height)
        x2 = int((x_center + width / 2) * image.width)
        y2 = int((y_center + height / 2) * image.height)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

    # 绘制 FP 框（蓝色）
    for box in fp_boxes:
        cls, x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image.width)
        y1 = int((y_center - height / 2) * image.height)
        x2 = int((x_center + width / 2) * image.width)
        y2 = int((y_center + height / 2) * image.height)
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)

    # 绘制文本
    draw.text((200, 100), f"pred_reliability: {pred_reliability:.4f}", fill="blue", font=font)
    draw.text((200, 200), f"gt_reliability: {gt_reliability:.4f}", fill="green", font=font)
    draw.text((200, 300), f"absolute_error: {absolute_error:.4f}", fill="red", font=font)

    return image

# 推理函数
def predict(model, test_loader, device, save_dir='./results_reliability/tp_fp_reliability'):
    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, (images, labels, original_images, reliability) in enumerate(test_loader):
        # 调整输入形状
        batch_size, num_images_per_sample, channels, height, width = images.shape
        images = images.view(-1, channels, height, width).to(device)  # [batch_size * num_images_per_sample, channels, height, width]
        original_images = original_images.to(device)

        # 推理
        with torch.no_grad():
            classification_output, regression_output = model(images, original_images)

        # 获取预测结果
        _, preds = torch.max(classification_output, 1)
        pred_reliability = regression_output.mean().item()  # 预测的可靠性
        gt_reliability = reliability.mean().item()  # 真实的可靠性
        absolute_error = abs(pred_reliability - gt_reliability)  # 绝对误差

        # 获取 TP 和 FP 框
        tp_boxes = []
        fp_boxes = []
        for i, pred in enumerate(preds):
            if pred == 0:  # TP
                # 构建正确的文件路径
                image_name = test_loader.dataset.image_names[batch_idx]  # 当前图像名称
                predict_file_path = os.path.join(test_loader.dataset.predict_dir, image_name.replace('.jpg', '.txt'))
                tp_boxes.extend(test_loader.dataset.parse_yolo_file(predict_file_path))
            else:  # FP
                # 构建正确的文件路径
                image_name = test_loader.dataset.image_names[batch_idx]  # 当前图像名称
                predict_file_path = os.path.join(test_loader.dataset.predict_dir, image_name.replace('.jpg', '.txt'))
                fp_boxes.extend(test_loader.dataset.parse_yolo_file(predict_file_path))

        # 将原始图像转换为 PIL 图像
        original_image = transforms.ToPILImage()(original_images.squeeze().cpu())

        # 在原图上绘制 TP 和 FP 框以及文本
        result_image = draw_boxes_and_text(original_image, tp_boxes, fp_boxes, pred_reliability, gt_reliability, absolute_error)

        # 保存结果图像
        image_name = test_loader.dataset.image_names[batch_idx]  # 当前图像名称
        result_image.save(os.path.join(save_dir, image_name))
        print(f"Saved result for image: {image_name}")


# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试集
    data_dir = './dataset/voc2069'
    test_dataset = VOCDataset(
        image_dir=os.path.join(data_dir, 'JPEGImages'),
        predict_dir=os.path.join(data_dir, 'predict_labels'),
        label_dir=os.path.join(data_dir, 'labels'),
        image_set_file=os.path.join(data_dir, 'ImageSets/Main/test.txt'),
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # 加载最佳模型
    best_model_path = './model_realiablity/best_model.pth'
    model = MultiTaskModel().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 进行推理
    predict(model, test_loader, device)

if __name__ == '__main__':
    main()
"""
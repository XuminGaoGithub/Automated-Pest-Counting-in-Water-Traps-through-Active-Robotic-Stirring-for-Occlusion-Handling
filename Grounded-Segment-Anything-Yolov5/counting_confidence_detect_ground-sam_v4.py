'''
python counting_confidence_detect_ground-sam_v4.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /home/xumin/Open-GroundingDino/weights/groundingdino_swint_ogc.pth --sam_hq_checkpoint /home/newdrive/summerschool/Grounded-Segment-Anything/sam_hq_vit_b.pth --use_sam_hq  --input_dir ./counting_confidence/src --output_dir "./simulation_output/image_processing" --box_threshold 0.5 --text_threshold 0.25 --device "cuda" --weights /home/newdrive/Phd/Modification_yolov5/runs/train/voc2060_yolov5s-ODConv-cotnet2/weights/best.pt --device_yolo 0 --save-txt --save-conf --source ./counting_confidence/yellow_pan

(1)
#parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to SAM-HQ checkpoint file")
#--sam_hq_checkpoint: python counting_confidence_grounded_sam.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /home/xumin/Open-GroundingDino/weights/groundingdino_swint_ogc.pth --sam_hq_checkpoint sam_hq_vit_b.pth --use_sam_hq  --input_dir /home/newdrive/summerschool/Grounded-Segment-Anything/counting_confidence/src --output_dir "./simulation_output/image_processing" --box_threshold 0.5 --text_threshold 0.25 --device "cuda"

#parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to SAM checkpoint file")
#--sam_checkpoint: python counting_confidence_grounded_sam.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /home/xumin/Open-GroundingDino/weights/groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_b_01ec64.pth  --input_dir /home/newdrive/summerschool/Grounded-Segment-Anything/counting_confidence/src --output_dir "./simulation_output/image_processing" --box_threshold 0.5 --text_threshold 0.25 --device "cuda"

(2)
#input_image for input_image without resize
#mask = cv2.erode(mask, kernel_erode, iterations=2)

#input_image for input_image with resize
#mask = cv2.erode(mask, kernel_erode, iterations=1)
'''


# python counting_confidence_detect_ground-sam_v4.py在v3版本上添加了perception<->speed部分

'''
0.#save_output_with_opencv_boxes_(frame,yellow_pan_boxes,yellow_pan_phrases,yellow_pan_output_image_path)
1.#save_mask_data(args.output_dir, masks, filtered_boxes_crop, filtered_phrases, image_name)
2.所有非必要#cv2.imwrite -> ##cv2.imwrite
3.所有非必要plt.savefig->#plt.savefig
    3.1
    """
        # 可视化梯度图像
        grad_magnitude_display = cv2.convertScaleAbs(magnitude)
    
        # 保存梯度图像
        #cv2.imwrite(output_image_path, grad_magnitude_display)
    """
    3.2
    #self._visualize_all_results()

4.#img, mask_stirring_tool = save_output_with_opencv(image_cv, largest_mask, filtered_boxes_crop, filtered_phrases, output_image_path)
5.comment 各个处理节点run_time计算
6.这里为了加速，只单独保存Adaptive_DBSCN聚类算法的聚类程度得分
    def _run_all_clustering(self):
        """执行所有聚类方法"""
        self.cluster_results = {
            'Adaptive DBSCAN': self._dbscan_clustering() #,
            #'MeanShift': self._mean_shift_clustering(),
            #'GMM': self._gmm_clustering(),
            #'KMeans': self._kmeans_clustering(),
            #'Hierarchical': self._hierarchical_clustering(),
            #'Spectral': self._spectral_clustering(),
            #'HDBSCAN': self._hdbscan_clustering()
        }

#with open("./simulation_output/score/result.txt", "a") as f:
                #for method, score in self.cluster_scores.items():
                    #f.write(f"{self.image_name + '.jpg'},{method},{score:.2f}\n")
# 即使没有聚类，也将结果写入文件，score 设置为 0
#with open("./simulation_output/score/result.txt", "a") as f:
                #f.write(f"{self.image_name + '.jpg'},'Adaptive DBSCAN',0.00\n")
                
7.comment drawContours and cv2.circle
#cv2.drawContours(mask_with_contours, contours, -1, (0, 0, 255), 2)  # 绿色轮廓
#cv2.drawContours(mask_with_contours_filter_noise, [contour], -1, 0, thickness=cv2.FILLED)  # 填充为黑色
#cv2.drawContours(mask_with_contours_filter_noise, filtered_contours_1, -1, (0, 0, 255), 2)  # 绿色轮廓
#cv2.circle(mask_with_contours_filter_noise, (cx, cy), 5, (0, 0, 255), -1)
#cv2.drawContours(mask_with_contours_filter_noise, filtered_contours_1, -1, (0, 0, 255), 2)  # 绿色轮廓
#cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)
#cv2.drawContours(after_filtered_contour_mask_with_centroids, filtered_contours, -1, (0, 0, 255), 2)
#cv2.drawContours(after_filtered_contour_mask_with_centroids, filtered_contours, -1, (0, 0, 255), 2)      
#for cx, cy in filtered_centroids:
            #cv2.circle(after_filtered_contour_mask_with_centroids, (cx, cy), 5, (0, 0, 255), -1)
 
#for cx, cy in filtered_centroids:
            #cv2.circle(only_centroids_mask, (cx, cy), 5, (0, 0, 255), -1)

8. 采用轻量化版本yolov5
'''


import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image
import torch

from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate


import os,time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, MeanShift, KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
import hdbscan
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score
import warnings
from torchvision.ops import box_convert

#from counting_confidence_detect_2 import detect_on_image
from utils.augmentations import letterbox  # Add this import at the top
import os
import time

# YOLOV5部分
import argparse
import os,time
import platform
import sys
from pathlib import Path
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import time


# YOLOv5 root directory
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadFromImage
from utils.general import (
    LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
    increment_path, non_max_suppression, soft_nms, print_args, scale_coords, strip_optimizer, xyxy2xywh
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
#from sklearn.metrics import plot_confusion_matrix
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou


import torch
from pyiqa import create_metric
import cv2
from PIL import Image

warnings.simplefilter(action='ignore', category=FutureWarning)
# 创建结果目录
os.makedirs("./simulation_output/analysis", exist_ok=True)
os.makedirs("./simulation_output/score", exist_ok=True)


def calculate_entropy(image):
    """计算图像熵"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # 归一化
    entropy = -np.sum(hist * np.log2(hist + 1e-10))  # 避免 log(0)
    return entropy


# https://github.com/chaofengc/IQA-PyTorch/tree/main
#如何配置iqa
# pip install pyiqa
# pip install timm==0.6.7
def calculate_niqe(iqa_model, img_rgb, device=None):
    # Set device if not specified
    #if device is None:
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'  # Use cpu to save GPU memory
    #device = 'cuda' #Could use cpu to save GPU memory
    # Create NIQE metric model (No-Reference metric)
    #iqa_model = create_metric('niqe', metric_mode='NR', device=device)

    # Load image using OpenCV and convert to PIL Image (pyiqa expects PIL Image)
    img_pil = Image.fromarray(img_rgb)

    # Calculate NIQE score
    score = iqa_model(img_pil).cpu().item()

    return score


def compute_gradients_entropy_niqe(image, iqa_model, output_image_path, output_blurriness_path):
    # 读取图像
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算x和y方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的模
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 计算模糊度（这里取平均梯度模作为模糊度的简单表示）
    blurriness = np.mean(magnitude)


    """
    # 可视化梯度图像
    grad_magnitude_display = cv2.convertScaleAbs(magnitude)

    # 保存梯度图像
    #cv2.imwrite(output_image_path, grad_magnitude_display)

    # 绘制模糊度图像并保存
    plt.figure(figsize=(6, 6))
    plt.imshow(grad_magnitude_display, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.colorbar()
    plt.savefig(output_blurriness_path)
    plt.close()
    """


    """计算图像熵"""
    entropy = calculate_entropy(gray)

    """计算图像质量"""
    niqe_score=calculate_niqe(iqa_model,img_rgb)
    #print('niqe_score:',niqe_score)
    #niqe_score = 0

    return blurriness,entropy,niqe_score


# Add this new function to handle direct image input
def detect_on_image(crop_img, model, stride, names, pt, device='0', imgsz=(640, 640), conf_thres=0.5, iou_thres=0.6):
    # Initialize device
    device = select_device(device)

    # Load model
    #model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=False)
    #stride, names, pt = model.stride, model.names, model.pt
    #imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Convert image to tensor
    im = letterbox(crop_img, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0-255 to 0.0-1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    with torch.no_grad():
        pred = model(im, augment=False, visualize=False)

    # Apply NMS
    #pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
    # Or use soft_nms if preferred:
    pred = soft_nms(pred, conf_thres, iou_thres, multi_label=True)

    return pred  # Return predictions for first (and only) image in batch





def calculate_aphid_size_by_pred(pred):
    """
    根据预测结果直接计算蚜虫的平均大小（基于检测框的宽度和高度）
    :param pred: YOLOv5预测结果，格式为tensor([[x1, y1, x2, y2, conf, class], ...])
    :return: 蚜虫的平均大小（像素）
    """
    # 如果没有检测框，返回默认值
    if pred is None or len(pred) == 0:
        print("No bounding boxes found in prediction results")
        return 10.0  # 返回一个默认值

    # 提取所有检测框的坐标 (x1, y1, x2, y2)
    boxes = pred[:, :4]


    # 计算每个检测框的宽度和高度
    widths = boxes[:, 2] - boxes[:, 0]  # x2 - x1
    heights = boxes[:, 3] - boxes[:, 1]  # y2 - y1


    # 计算平均宽度和高度
    avg_width = widths.float().mean().item()
    avg_height = heights.float().mean().item()

    # 返回蚜虫的平均大小（像素）
    return (avg_width + avg_height) / 2

# ================== 参数自动设置模块 ==================
def calculate_aphid_size(image_name, labels_dir, img_width, img_height):
    """
    计算蚜虫的平均大小（基于yolo格式的预测结果txt）
    :param image_name: 图像名称（不带扩展名）
    :param labels_dir: YOLO 格式的标签文件目录
    :param img_width: 图像的宽度（像素）
    :param img_height: 图像的高度（像素）
    :return: 蚜虫的平均大小（像素）
    """
    # 构建对应的标签文件路径
    label_path = os.path.join(labels_dir, f"{image_name}.txt")

    # 如果标签文件不存在，返回默认值（例如 10px）
    if not os.path.exists(label_path):
        print(f"Prediction label file not found: {label_path}")
        return 10.0  # 返回一个默认值

    # 读取标签文件
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 解析 YOLO 格式的检测框
    widths = []
    heights = []
    for line in lines:
        # YOLO 格式：class_id x_center y_center width height confidence
        data = line.strip().split()
        if len(data) < 5:  # 确保每行至少有 5 个值
            continue

        # 解析宽度和高度（归一化值）
        width_norm = float(data[3])  # 宽度（归一化）
        height_norm = float(data[4])  # 高度（归一化）

        # 将归一化的宽度和高度映射到像素值
        width = width_norm * img_width
        height = height_norm * img_height

        widths.append(width)
        heights.append(height)

        #print('widths,heights:',widths,heights)

    # 如果没有检测框，返回默认值
    if len(widths) == 0:
        print(f"No bounding boxes found in {label_path}")
        return 10.0  # 返回一个默认值

    # 计算平均宽度和高度
    avg_width = np.mean(widths)
    avg_height = np.mean(heights)

    # 返回蚜虫的平均大小（像素）
    return (avg_width + avg_height) / 2


def auto_set_parameters(aphid_size,centroids, img_width, img_height, image_name, labels_dir):
    """自动设置所有聚类相关参数"""
    # 计算蚜虫平均大小
    #print('image_name:', image_name)
    #print('labels_dir:', labels_dir)
    #print('w:', img_width)
    #print('h:', img_height)
    #aphid_size = calculate_aphid_size(image_name, labels_dir, img_width, img_height)
    threshold = 1.5 * aphid_size  # 动态阈值
    #print('avg_aphid_size:', aphid_size)

    # 计算最近邻距离
    dist_matrix = distance_matrix(centroids, centroids)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_distances = dist_matrix.min(axis=1)

    mean_nearest_dist = np.mean(nearest_distances)
    std_nearest_dist = np.std(nearest_distances)

    # 如果没有聚类现象
    if mean_nearest_dist > threshold:
        return {
            'no_clustering': True,
            'radius': mean_nearest_dist * 1.5  # 设置一个默认的 radius
        }

    eps = aphid_size * 1
    radius = eps * 2
    min_samples = 2

    return {
        'avg_aphid_size': aphid_size,
        'eps': eps,
        'radius': radius,
        'min_samples': min_samples,
        'mean_shift_bandwidth': eps * 1,
        'kmeans_clusters': auto_set_kmeans_clusters(centroids),
        'gmm_components': auto_set_gmm_components(centroids),
        'hdbscan_min_cluster': min_samples,
        'no_clustering': False
    }


def auto_set_kmeans_clusters(centroids, max_clusters=10):
    """使用肘部法则自动设置 K-Means 簇数"""
    sse = []
    max_clusters = min(max_clusters, len(centroids) - 1)  # 防止溢出
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k).fit(centroids)
        sse.append(kmeans.inertia_)

    if len(sse) < 3:
        # 如果数据不足，返回默认值
        return len(sse)  # 或者返回 1

    # 找到肘部点（二阶差分最大值）
    elbow_point = np.argmax(np.diff(sse, 2)) + 1
    return elbow_point


def auto_set_gmm_components(centroids, max_components=10):
    """使用 BIC 自动设置 GMM 分量数"""
    bic = []

    # 确保 max_components 不超过样本数 - 1
    max_components = min(max_components, len(centroids) - 1)

    # 如果样本数不足，返回默认值
    if len(centroids) < 2:
        return 1

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n).fit(centroids)
        bic.append(gmm.bic(centroids))

    # 找到 BIC 最小值
    best_n = np.argmin(bic) + 1
    return best_n


# ================== 聚类分析模块 ==================
class AphidAnalyzer:
    def __init__(self, aphid_size,centroids, img_size, image_name, labels_dir):
        #self.aphid_size = aphid_size
        self.centroids = centroids
        self.img_width, self.img_height = img_size
        self.image_name = image_name
        self.labels_dir = labels_dir
        self.params = auto_set_parameters(aphid_size, centroids, self.img_width, self.img_height, self.image_name, self.labels_dir)

    def analyze_distribution(self):
        """执行完整分析流程"""
        self._basic_analysis()
        if not self.params['no_clustering']:
            self._run_all_clustering()
            #self._visualize_all_results()

            # 将评分结果保存到 result.txt 文件
            #with open("./simulation_output/score/result.txt", "a") as f:
                #for method, score in self.cluster_scores.items():
                    #f.write(f"{self.image_name + '.jpg'},{method},{score:.2f}\n")

            # 只返回Adaptive DBSCAN计算得到的score
            return self.cluster_scores.get('Adaptive DBSCAN', 0.0)

        else:
            # 即使没有聚类，也将结果写入文件，score 设置为 0
            #with open("./simulation_output/score/result.txt", "a") as f:
                #f.write(f"{self.image_name + '.jpg'},'Adaptive DBSCAN',0.00\n")
            print("No clustering detected, skipping clustering step.")

            # 只返回Adaptive DBSCAN计算得到的score
            return 0.0


    def _basic_analysis(self):
        """基础统计分析"""
        #print('self.centroids:',self.centroids)
        self.nearest_dist, self.local_density = analyze_nearest_neighbors(
            self.centroids, self.params['radius'])

        print("\n===== Basic Statistics =====")
        print(f"Total aphids: {len(self.centroids)}")
        print(f"Average nearest neighbor distance: {np.mean(self.nearest_dist):.2f} px")
        print(f"Maximum local density: {np.max(self.local_density):.16f}/px²")

    def calculate_clustering_score(self, cluster_labels, centroids, epsilon=1e-6):
        """
        计算聚群程度评分
        :param cluster_labels: 聚类标签数组（如 DBSCAN 的 labels_）
        :param centroids: 蚜虫的中心点坐标
        :param epsilon: 用于避免除零错误的小常数
        :return: 聚群程度评分
        """
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # 去除噪声点
            print('remove noise')

        N = len(unique_labels)  # 聚类数量
        print('Clustering number_N:', N)

        if not unique_labels:
            return 0  # 如果没有聚类，返回 0

        score = 0  # 初始化评分

        # 计算每个聚群的贡献
        for label in unique_labels:
            points = centroids[cluster_labels == label]
            num_points = len(points)  # 当前聚群的数量
            print('Clustering number of' + ' ' + str(label) + ':', num_points)

            if num_points > 1:
                # 计算当前聚群的平均距离
                dist_matrix = distance_matrix(points, points)
                np.fill_diagonal(dist_matrix, np.inf)
                avg_distance = np.mean(dist_matrix.min(axis=1))
            else:
                avg_distance = epsilon  # 如果只有一个点，平均距离设为 epsilon

            # 避免除零错误
            if avg_distance == 0:
                avg_distance = epsilon

            print('Avg_distance of' + ' Clustering' + ' ' + str(label) + ':', avg_distance)

            # 计算当前聚群的贡献并累加到总评分
            score += num_points * (1 / avg_distance)

        return score

    def _run_all_clustering(self):
        """执行所有聚类方法"""
        self.cluster_results = {
            'Adaptive DBSCAN': self._dbscan_clustering() #,
            #'MeanShift': self._mean_shift_clustering(),
            #'GMM': self._gmm_clustering(),
            #'KMeans': self._kmeans_clustering(),
            #'Hierarchical': self._hierarchical_clustering(),
            #'Spectral': self._spectral_clustering(),
            #'HDBSCAN': self._hdbscan_clustering()
        }

        # 计算每种聚类方法的评分
        self.cluster_scores = {}
        for method, labels in self.cluster_results.items():
            score = self.calculate_clustering_score(labels, self.centroids)  # 确保传递的是 labels
            self.cluster_scores[method] = score
            print(f"{method} clustering degree score: {score:.2f}")

    # ========== 各聚类方法实现 ==========
    def _dbscan_clustering(self):
        clustering = DBSCAN(eps=self.params['eps'],
                            min_samples=self.params['min_samples']).fit(self.centroids)
        return clustering.labels_

    def _mean_shift_clustering(self):
        clustering = MeanShift(bandwidth=self.params['mean_shift_bandwidth']).fit(self.centroids)
        return clustering.labels_

    def _gmm_clustering(self):
        gmm = GaussianMixture(n_components=self.params['gmm_components']).fit(self.centroids)
        return gmm.predict(self.centroids)

    def _kmeans_clustering(self):
        kmeans = KMeans(n_clusters=self.params['kmeans_clusters']).fit(self.centroids)
        return kmeans.labels_

    def _hierarchical_clustering(self):
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.params['eps'] * 2
        ).fit(self.centroids)
        return clustering.labels_

    def _spectral_clustering(self):
        n_neighbors = min(len(self.centroids), 10)  # n_neighbors should be <= n_samples
        clustering = SpectralClustering(
            n_clusters=self.params['kmeans_clusters'],
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors
        ).fit(self.centroids)
        return clustering.labels_

    def _hdbscan_clustering(self):
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=self.params['hdbscan_min_cluster']
        ).fit(self.centroids)
        return clustering.labels_

    # ========== 可视化模块 ==========
    def _visualize_all_results(self):
        """生成所有可视化结果"""
        self._plot_density_heatmap()
        for method, labels in self.cluster_results.items():
            self._plot_cluster_results(method, labels)
            self._plot_cluster_heatmaps(method, labels)

    def _plot_density_heatmap(self):
        """绘制密度热图"""
        heatmap = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        for (x, y), density in zip(self.centroids, self.local_density):
            x, y = int(x), int(y)
            heatmap[y, x] = density
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)


        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Local Density (aphids/px²)")
        plt.title("Local Density Heatmap")
        plt.savefig(f"./simulation_output/analysis/{self.image_name}_density_heatmap.png")
        plt.close()


    def _plot_cluster_results(self, method, labels):
        """绘制聚类结果散点图"""
        unique_labels = set(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

        plt.figure(figsize=(self.img_width / 100, self.img_height / 100), dpi=100)
        plt.xlim(0, self.img_width)
        plt.ylim(0, self.img_height)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = "black"
            points = self.centroids[labels == label]
            plt.scatter(points[:, 0], points[:, 1],
                        color=color,
                        label=f"Cluster {label}" if label != -1 else "Noise",
                        s=80, alpha=0.7)


        plt.legend()
        plt.title(f"{method} Clustering Results")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.savefig(f"./simulation_output/analysis/{self.image_name}_{method}_clusters.png")
        plt.close()


    def _plot_cluster_heatmaps(self, method, labels):
        """生成聚类相关热图"""
        # 数量热图
        count_heatmap = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        # 距离热图
        dist_heatmap = np.zeros((self.img_height, self.img_width), dtype=np.float32)

        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue
            points = self.centroids[labels == label]

            # 计算簇内平均距离
            if len(points) > 1:
                dist_matrix = distance_matrix(points, points)
                np.fill_diagonal(dist_matrix, np.inf)
                avg_dist = np.mean(dist_matrix.min(axis=1))
            else:
                avg_dist = 0

            # 更新热图
            for (x, y) in points:
                x, y = int(x), int(y)
                count_heatmap[y, x] = len(points)
                dist_heatmap[y, x] = avg_dist

        # 应用高斯模糊
        count_heatmap = cv2.GaussianBlur(count_heatmap, (25, 25), 0)
        dist_heatmap = cv2.GaussianBlur(dist_heatmap, (25, 25), 0)


        # 绘制数量热图
        plt.figure(figsize=(12, 8))
        plt.imshow(count_heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Aphid Count per Cluster")
        plt.title(f"{method} - Aphid Count Heatmap")
        plt.savefig(f"./simulation_output/analysis/{self.image_name}_{method}_count_heatmap.png")
        plt.close()

        # 绘制距离热图
        plt.figure(figsize=(12, 8))
        plt.imshow(dist_heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Average Distance (px)")
        plt.title(f"{method} - Intra-cluster Distance Heatmap")
        plt.savefig(f"./simulation_output/analysis/{self.image_name}_{method}_distance_heatmap.png")
        plt.close()



# ================== 数据处理工具函数 ==================
def analyze_nearest_neighbors(centroids, radius):
    """分析最近邻距离和局部密度"""
    dist_matrix = distance_matrix(centroids, centroids)
    np.fill_diagonal(dist_matrix, np.inf)

    nearest_dist = dist_matrix.min(axis=1)
    local_density = [np.sum(row <= radius) / (np.pi * radius ** 2)
                     for row in dist_matrix]

    return nearest_dist, local_density



def load_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.to(device)
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    #model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrases
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases, logits, tokenized

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([50/255, 0, 100/255, 0.3])
    h, w = mask.shape[-2:]
    #print('h,w:',h,w)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)

#TODO: mask image size should be same to src image (use opencv to save)
def save_mask_data(output_dir, mask_list, box_list, label_list, image_name):
    value = 0
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1

    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')

    # 使用原图的名称来保存 mask
    mask_image_name = f"{os.path.splitext(image_name)[0]}_mask.jpg"
    plt.savefig(os.path.join(output_dir, mask_image_name), bbox_inches="tight", dpi=300, pad_inches=0.0)

    plt.close()  # 关闭当前的图像窗口，释放内存


def save_output_with_opencv(image_cv, largest_mask, filtered_boxes, filtered_phrases, output_image_path):
    copy_image_1 = image_cv.copy()

    # 绘制bounding boxes
    for box, label in zip(filtered_boxes, filtered_phrases):
        box = box.numpy().astype(int)

        # 防止label超出图像顶部，检查box顶部是否太靠近边缘
        if box[1] - 10 < 0:
            label_y = box[3] + 40  # 如果上边界接触图像边界，将label放在box的下方
        else:
            label_y = box[1] - 10  # 否则，将label放在box的上方

        # 绘制box
        cv2.rectangle(copy_image_1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 5)
        # 绘制label
        cv2.putText(copy_image_1, label, (box[0], label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # 保存带bounding box和label的图像
    copy_image_bgr_1 = cv2.cvtColor(copy_image_1, cv2.COLOR_RGB2BGR)
    #cv2.imwrite(output_image_path.replace('.jpg', '_bbox.jpg'), copy_image_bgr_1)

    # 将黄色mask叠加到原图上
    copy_image_2 = image_cv.copy()
    yellow_color = np.array([255, 255, 0], dtype=np.uint8)  # BGR格式的黄色
    mask_indices = np.squeeze(largest_mask)  # 将形状 (1, 2985, 2985) 变为 (2985, 2985)

    # 确保mask和图像的维度匹配
    #print('copy_image_2.shape:', copy_image_2.shape)
    #print('mask_indices.shape:', mask_indices.shape)

    # 叠加mask为黄色区域
    copy_image_2[mask_indices > 0] = yellow_color

    # 保存叠加黄色mask的图像
    copy_image_bgr_2 = cv2.cvtColor(copy_image_2, cv2.COLOR_RGB2BGR)
    #cv2.imwrite(output_image_path.replace('.jpg', '_yellow_mask.jpg'), copy_image_bgr_2)

    return copy_image_bgr_2,mask_indices


#正常处理
def save_output_with_opencv_boxes(original_frame, boxes, phrases, output_path):

    # 1. 输入验证
    if original_frame is None or original_frame.size == 0:
        raise ValueError("原始帧图像无效")
    if len(boxes) != len(phrases):
        raise ValueError("检测框和标签数量不匹配")

    # 2. 创建图像副本
    img_to_draw = original_frame.copy()
    h, w = img_to_draw.shape[:2]

    # 3. 处理每个检测框
    for box, label in zip(boxes, phrases):
        try:
            # 将相对坐标转换为绝对坐标
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()

            # 转换坐标: [cx, cy, width, height] -> [x1, y1, x2, y2]
            cx, cy, bw, bh = box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            # 验证坐标有效性
            if x1 >= x2 or y1 >= y2 or x2 > w or y2 > h:
                print(f"警告: 跳过无效box坐标 - x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
                continue

            # 计算标签位置 (避免超出图像边界)
            label_pos = (x1, max(y1 - 10, 10))

            # 绘制边界框 (蓝色，线宽5)
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (255, 0, 0), 5)

            # 绘制标签 (蓝色，字体大小1.2，线宽3)
            cv2.putText(img_to_draw, label, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        except Exception as e:
            print(f"处理box {box} 时出错: {str(e)}")
            continue

    # 4. 保存结果图像
    cv2.imwrite(output_path, img_to_draw)

#label超出了图像范围，这种情况下，请将lable和logits绘制到bbox内部的左上角
def save_output_with_opencv_boxes_(original_frame, boxes, phrases, output_path):

    # 1. 输入验证
    if original_frame is None or original_frame.size == 0:
        raise ValueError("原始帧图像无效")


    # 2. 创建图像副本
    img_to_draw = original_frame.copy()
    h, w = img_to_draw.shape[:2]

    # 3. 绘制参数配置
    box_color = (0, 0, 255)  # 红色边框 (BGR格式)
    text_color = (255, 255, 255)  # 白色文字
    bg_color = (0, 0, 255)  # 红色背景 (与边框同色)
    box_thickness = 2
    font_scale = 0.5  # 更小的字体大小
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_padding = 3  # 文字内边距

    # 4. 处理每个检测框
    for box, label in zip(boxes, phrases):
        try:
            # 将相对坐标转换为绝对坐标
            if isinstance(box, torch.Tensor):
                box = box.cpu().numpy()

            # 转换坐标: [cx, cy, width, height] -> [x1, y1, x2, y2]
            cx, cy, bw, bh = box
            x1 = max(0, int((cx - bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            x2 = min(w, int((cx + bw / 2) * w))
            y2 = min(h, int((cy + bh / 2) * h))

            # 验证坐标有效性
            if x1 >= x2 or y1 >= y2:
                print(f"Warning: skip invalid box coordinations - x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
                continue

            # 绘制边界框
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), box_color, box_thickness)

            # 准备显示文本 (标签 + 置信度)
            display_text = f"{label}"

            # 计算文本大小
            (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)

            # 计算文本位置（框内左上角）
            text_x = x1 + text_padding
            text_y = y1 + text_h + text_padding

            # 如果文本超出框底部，则向上移动
            if text_y > y2:
                text_y = y1 - text_padding
                if text_y < 0:  # 如果上方也超出，则放在框内最上方
                    text_y = y1 + text_h

            # 如果文本超出框右侧，则调整文本
            if text_x + text_w > x2:
                display_text = f"{label.split()[0]}:{logit:.2f}"  # 缩短标签
                (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)

            # 绘制文本背景（增强可读性）
            cv2.rectangle(img_to_draw,
                          (text_x - text_padding, text_y - text_h - text_padding),
                          (text_x + text_w + text_padding, text_y + text_padding),
                          bg_color, -1)  # 填充矩形

            # 绘制文本
            cv2.putText(img_to_draw, display_text, (text_x, text_y),
                        font, font_scale, text_color, font_thickness)

        except Exception as e:
            print(f"Processing box {box} failed: {str(e)}")
            continue

    # 5. 保存结果图像
    cv2.imwrite(output_path, img_to_draw)



#图像处理部分#
def analysis(img, image_name, pred, output_folder_color, output_folder_contour,output_folder_heatmap, output_folder_gradient,
             kernel_dilate, kernel_erode,mask_stirring_tool):


    # Clarity, and density distribution analysis
    if img is not None:


        """ 只计算yellow pan区域的梯度模"""
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #print('type(mask_stirring_tool):', type(mask_stirring_tool))
        #print('mask_stirring_tool:',mask_stirring_tool)
        if mask_stirring_tool is not None:
            # 使用膨胀操作扩展mask边界3个像素
            kernel = np.ones((3, 3), np.uint8)  # 3x3内核表示扩展3个像素
            dilated_mask = cv2.dilate(mask_stirring_tool.astype(np.uint8), kernel, iterations=1)
            # 创建布尔掩码，非mask区域为True，膨胀后的mask区域为False
            non_mask_area = dilated_mask == 0

            # 创建布尔掩码，非mask区域为True，mask区域为False
            #non_mask_area = mask_stirring_tool == 0
            # 计算x和y方向的梯度，仅在非mask区域上进行计算
            #grad_x = np.zeros_like(gray, dtype=np.float64)
            #grad_y = np.zeros_like(gray, dtype=np.float64)
            #grad_x[non_mask_area] = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)[non_mask_area]
            #grad_y[non_mask_area] = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)[non_mask_area]
            # 计算梯度的模，仅限非mask区域
            #magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            # 计算模糊度（取非mask区域的平均梯度模）
            #blurriness = np.mean(magnitude[non_mask_area])
            # 可视化梯度图像
            #grad_magnitude_display = cv2.convertScaleAbs(magnitude)
            # 保存梯度图像
            #gradient_output_path = os.path.join(output_folder_gradient, f'gradient_{image_name}')
            ###cv2.imwrite(gradient_output_path, grad_magnitude_display)

        else:
            print('')
            # 计算x和y方向的梯度
            #grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            #grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            # 计算梯度的模
            #magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            # 计算模糊度（这里取平均梯度模作为模糊度的简单表示）
            #blurriness = np.mean(magnitude)
            # 可视化梯度图像
            #grad_magnitude_display = cv2.convertScaleAbs(magnitude)
            # 保存梯度图像
            #gradient_output_path = os.path.join(output_folder_gradient, f'gradient_{image_name}')
            ###cv2.imwrite(gradient_output_path, grad_magnitude_display)
        """



        # 计算图像的中心点
        h, w = img.shape[:2]
        #image_center = (w // 2, h // 2)

        # 颜色空间转换与阈值分割（生成mask）
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        inverted_mask = cv2.bitwise_not(mask)

        output_contour_mask_path = os.path.join(output_folder_contour, f'original_color_segmented_mask_{image_name}')
        #cv2.imwrite(output_contour_mask_path, inverted_mask)

        # 颜色分割结果：去除黄色背景
        result_color = cv2.bitwise_and(img, img, mask=inverted_mask)
        # 保存 original_color_segmented 图像
        original_color_segmented_path = os.path.join(output_folder_color, f'original_color_segmented_{image_name}')
        #cv2.imwrite(original_color_segmented_path, result_color)


        # 膨胀和腐蚀处理
        mask=inverted_mask
        #mask = cv2.dilate(inverted_mask, kernel_dilate, iterations=2)
        #mask = cv2.erode(mask, kernel_erode, iterations=2)
        mask = cv2.erode(mask, kernel_erode, iterations=1)

        #inverted_mask = cv2.bitwise_not(mask)

        # 保存原始 mask 图像
        original_mask_path = os.path.join(output_folder_contour, f'original_morphology_mask_{image_name}')
        #cv2.imwrite(original_mask_path, mask)



        # 形态学处理，闭运算（膨胀+腐蚀）连接靠近的物体
        # morph_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, kernel_dilate)
        morph_mask = mask

        # 查找轮廓并计算质心
        contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        mask_with_contours = np.stack([morph_mask] * 3, axis=-1)  # 将单通道mask扩展为3通道以绘制轮廓和质心

        # 绘制轮廓到mask图像和原图
        #cv2.drawContours(mask_with_contours, contours, -1, (0, 0, 255), 2)  # 绿色轮廓

        # 保存轮廓和质心的原图和mask图
        output_contour_mask_path = os.path.join(output_folder_contour, f'original_contour_mask_before_filter_noise_{image_name}')
        #cv2.imwrite(output_contour_mask_path, mask_with_contours)



        # 计算蚜虫的平均大小
        mask_with_contours_filter_noise = np.stack([morph_mask] * 3, axis=-1)  #
        image_name_without_ext = os.path.splitext(image_name)[0]  # 去掉 .jpg 后缀

        aphid_size = calculate_aphid_size_by_pred(pred[0])  # pred[0] 是包含检测框的tensor
        #print('aphid_size:', aphid_size)

        min_contour_area = (aphid_size * aphid_size) / 10  # 最小轮廓面积为 aphid_size 的 1/10
        #min_contour_area = (aphid_size * aphid_size) / 20  # 最小轮廓面积为 aphid_size 的 1/20

        #print('min_contour_area:', min_contour_area)

        # 过滤掉面积小于 min_contour_area 的轮廓
        filtered_contours_1 = []
        for contour in contours:
            # 计算轮廓的最小外接矩形面积
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            contour_area = w_rect * h_rect
            #print('w_rect,h_rect:',w_rect,h_rect)
            #print('contour_area:', contour_area)
            #print('min_contour_area:', min_contour_area)

            # 如果轮廓面积小于 min_contour_area，跳过该轮廓
            if contour_area < min_contour_area:
                #print(f"Contour at {image_name} is too small (area: {contour_area}), removing...")
                #cv2.drawContours(mask_with_contours_filter_noise, [contour], -1, 0, thickness=cv2.FILLED)  # 填充为黑色
                continue

            # 保留符合条件的轮廓
            filtered_contours_1.append(contour)

        # 绘制轮廓到mask图像和原图
        #cv2.drawContours(mask_with_contours_filter_noise, filtered_contours_1, -1, (0, 0, 255), 2)  # 绿色轮廓

        # 保存轮廓和质心的原图和mask图
        output_contour_mask_path = os.path.join(output_folder_contour, f'original_contour_mask_after_filter_noise_{image_name}')
        #cv2.imwrite(output_contour_mask_path, mask_with_contours_filter_noise)

        for contour in filtered_contours_1:
            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                # 在mask图上画质心（蓝色）
                #cv2.circle(mask_with_contours_filter_noise, (cx, cy), 5, (0, 0, 255), -1)

        # 绘制轮廓到mask图像和原图
        #cv2.drawContours(mask_with_contours_filter_noise, filtered_contours_1, -1, (0, 0, 255), 2)  # 绿色轮廓

        left_border = 0
        right_border = w
        top_border = 0
        bottom_border = h

        # 过滤角点背景
        filtered_contours = []
        filtered_centroids = []
        for contour in filtered_contours_1:
            # 获取轮廓的最小外接矩形
            x, y, w_rect, h_rect = cv2.boundingRect(contour)

            # 判断轮廓是否接触图像的两个边界
            touches_left = x == left_border
            touches_right = (x + w_rect) == right_border
            touches_top = y == top_border
            touches_bottom = (y + h_rect) == bottom_border

            # 检查轮廓是否同时接触两个边界
            touches_two_sides = (
                    (touches_left and touches_top) or
                    (touches_left and touches_bottom) or
                    (touches_right and touches_top) or
                    (touches_right and touches_bottom)
            )

            # 如果接触了两个边界，则将该轮廓视为背景并移除
            if touches_two_sides:
                print(f'Contour at {image_name} detected as background (touching two borders), removing...')
                continue  # 移除接触边界的轮廓

            filtered_contours.append(contour)  # 保留不接触角的轮廓
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                filtered_centroids.append((cx, cy))

        # 重新生成过滤后的mask图
        filtered_mask = np.zeros_like(mask)
        #cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)



        # 保存 after_filtered_contour_mask 图像，带质心
        after_filtered_contour_mask_with_centroids = np.stack([filtered_mask] * 3, axis=-1)
        #cv2.drawContours(after_filtered_contour_mask_with_centroids, filtered_contours, -1, (0, 0, 255), 2)
        after_filtered_contour_mask_path = os.path.join(output_folder_contour,
                                                        f'after_filter_corner_contour_mask_{image_name}')
        #cv2.imwrite(after_filtered_contour_mask_path, after_filtered_contour_mask_with_centroids)

        # 保存 after_filtered_contour_mask 图像，带质心
        after_filtered_contour_mask_with_centroids = np.stack([filtered_mask] * 3, axis=-1)
        #cv2.drawContours(after_filtered_contour_mask_with_centroids, filtered_contours, -1, (0, 0, 255), 2)

        #for cx, cy in filtered_centroids:
            #cv2.circle(after_filtered_contour_mask_with_centroids, (cx, cy), 5, (0, 0, 255), -1)

        after_filtered_contour_mask_path = os.path.join(output_folder_contour, f'after_filter_corner_contour_mask_with_centroids_{image_name}')
        #cv2.imwrite(after_filtered_contour_mask_path, after_filtered_contour_mask_with_centroids)

        filtered_mask = np.zeros_like(mask)
        only_centroids_mask = np.stack([filtered_mask] * 3, axis=-1)
        #for cx, cy in filtered_centroids:
            #cv2.circle(only_centroids_mask, (cx, cy), 5, (0, 0, 255), -1)
        after_filtered_contour_mask_path = os.path.join(output_folder_contour,
                                                        f'only_centroids_{image_name}')
        #cv2.imwrite(after_filtered_contour_mask_path, only_centroids_mask)
        #print('filtered_centroids:',filtered_centroids)

    return filtered_centroids,aphid_size


def resize_frame(frame):
    """按照最短边缩放到640的比例来resize图像"""
    # 获取原始尺寸
    height, width = frame.shape[:2]
    # 计算缩放比例（将最短边缩放到 640）
    short_side = min(height, width)
    scale = 640.0 / short_side
    # 计算新的尺寸
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    # 缩放图像
    return cv2.resize(frame, (new_width, new_height))


def load_image_from_frame(frame):
    """直接从视频帧加载图像并转换为需要的格式"""
    # 将OpenCV BGR格式转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 转换为PIL图像
    image_pil = Image.fromarray(frame_rgb)

    # 转换处理
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image, _ = transform(image_pil, None)
    return image_pil, image


# ====== 核心函数 ======
def compute_delta_total(sliding_window, weights,negatives):
    deltas = []
    for key in weights:
        values = [item[key] for item in sliding_window]
        # 线性变化（不取绝对值）
        delta = sum((values[i+1] - values[i]) for i in range(len(values) - 1)) / (len(values) - 1)
        if key in negatives:
            delta *= -1  # 越小越好 → 变化越大越差，反向处理
        deltas.append(weights[key] * delta)
    return sum(deltas)

def process_directory(input_dir, output_dir, args):
    output_folder_src = './counting_confidence/src'  # './result'
    output_folder_yellow_pan = './counting_confidence/yellow_pan'  # './result'
    output_folder_color = './simulation_output/image_processing' #'./result'
    output_folder_contour = './simulation_output/image_processing' #'./result'
    output_folder_heatmap = './simulation_output/image_processing' #'./result'
    output_folder_gradient = './simulation_output/image_processing' #'./result'
    output_folder_counting_confidence = './simulation_output/score'

    os.makedirs(output_folder_src, exist_ok=True)
    os.makedirs(output_folder_yellow_pan, exist_ok=True)
    os.makedirs(output_folder_color, exist_ok=True)
    os.makedirs(output_folder_contour, exist_ok=True)
    os.makedirs(output_folder_heatmap, exist_ok=True)
    os.makedirs(output_folder_heatmap, exist_ok=True)
    os.makedirs(output_folder_counting_confidence, exist_ok=True)


    ## Initialize YOLOv5 detection components (one-time setup) ##
    nc = 1
    # confusion_matrix = ConfusionMatrix(nc=nc)
    source = str(args.source)
    save_img = not args.nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir = Path(args.project) / args.name  # 每次都使用相同路径，不自增
    (save_dir / 'labels' if args.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device_yolo = select_device(args.device_yolo)
    model_yolo = DetectMultiBackend(args.weights, device=device_yolo, dnn=args.dnn, data=args.data, fp16=args.half)
    stride, names, pt = model_yolo.stride, model_yolo.names, model_yolo.pt
    names = dict(enumerate(model_yolo.names if hasattr(model_yolo, 'names') else model_yolo.module.names))
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    #print('imgsz:',imgsz)
    # Run inference
    model_yolo.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    yellow_pan_dir = "./counting_confidence/yellow_pan"
    os.makedirs(yellow_pan_dir, exist_ok=True)
    blurriness_folder_path = './runs/blurriness/'  # 替换为你的文件夹路径
    os.makedirs(blurriness_folder_path, exist_ok=True)


    # 设置结构元素大小（膨胀和腐蚀的核）
    kernel_dilate = np.ones((7, 7), np.uint8)
    kernel_erode = np.ones((3, 3), np.uint8)


    device = torch.device(args.device)

    # Load Grounding DINO model
    model = load_model(args.config, args.grounded_checkpoint, device=device)

    #model = model.to(device)
    model = model.to(device).eval()

    # Initialize SAM
    if args.use_sam_hq:
        predictor = SamPredictor(
            sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(device))


    #device_iqa = 'cpu'  # Use cpu to save GPU memory
    device_iqa = 'cuda' #Could use cpu to save GPU memory
    # Create NIQE metric model (No-Reference metric)
    iqa_model = create_metric('niqe', metric_mode='NR', device=device_iqa)


############################################################################
    # ====== 权重定义：根据归一化 R² =========
    r2_scores = {
        "PDU": 0.1135,
        "MeanConfidence": 0.1524,
        "Blurriness": 0.1737,
        "IC": 0.2559,
        "IQA": 0.2722,
        "Detections": 0.2869
    }

    total_r2 = sum(r2_scores.values())
    weights = {k: v / total_r2 for k, v in r2_scores.items()}
    print("=== R²归一化权重 ===")
    for k, w in weights.items():
        print(f"{k}: {w:.4f}")

    # ====== 参数 ======
    negatives = ['IQA', 'IC', 'PDU', 'Detections']  # 越小越好
    k = 3
    speed = 0.5
    #adjust_factor = 0.2  # 控制调节幅度（建议 0.1~0.3）
    adjust_factor = 0.1  # 控制调节幅度（建议 0.1~0.3）
    stop_threshold = 0.05
    max_speed = 1.0
    min_speed = 0.0

    speed_records=[]
    change_total = []

    # ======数据记录容器 ======
    history = []  # 每秒记录一组 dict 数据
###########################################################################



    # 改为0为摄像头，或者写入视频路径例如 "video.mp4"
    video_source = "/home/newdrive/simulation/test/test_20241120/video/Aphid_50.mp4"  # 或者 "your_video.mp4"

    # 提取视频文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(video_source))[0]
    print('base_name:',base_name)

    cap = cv2.VideoCapture(video_source)
    interval_sec = 1  # 间隔1秒
    time_records=[]
    frame_index = 0


    """读取每一帧"""
    """
    while True:
        start_time = time.time()
        # 1. 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 视频结束或摄像头读取失败，等待2秒后继续...")
            time.sleep(2)
            cap = cv2.VideoCapture(video_source)
            continue
    """


    #读间隔1s取每一帧
    while True:

        start_time = time.time()
        print('\n\n')
        # 计算下一帧的时间点（单位毫秒）
        #start_time_read = time.time()
        ms_pos = frame_index * interval_sec * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, ms_pos)
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 视频读取结束或失败。")
            break  # 不再重启，直接退出循环
        #end_time_read = time.time()
        #time_read = end_time_read-start_time_read
        #print("time_read:", time_read)


        # 1. 对图像进行resize
        #start_time_resize_1 = time.time()
        frame = resize_frame(frame)
        #start_time_resize_2 = time.time()
        #time_resize = start_time_resize_2-start_time_resize_1
        #print("time_resize:",time_resize)

        image_name = f"frame_{frame_index:06d}.jpg"
        print('image_name:',image_name)
        input_image_path = os.path.join(input_dir, image_name)
        print('input_image_path:',input_image_path)
        #cv2.imwrite(input_image_path, frame)

        # 2. 直接加载处理图像
        image_pil, image = load_image_from_frame(frame)
        image_source = np.asarray(image_pil)

        # Run Grounding DINO model
        #start_time_dino_1 = time.time()

        boxes_filt, pred_phrases, logits, tokenized = get_grounding_output(
            model, image, args.text_prompt, args.box_threshold, args.text_threshold, device=device
        )

        #start_time_dino_2 = time.time()
        #time_dino = start_time_dino_2 - start_time_dino_1
        #print("time_dino:", time_dino)


###------------ 1.yellow_pan detection by grounded_dino --------------------------###

        # 检查是否有 'yellow pan' 在检测结果中
        yellow_pan_boxes = []
        yellow_pan_phrases = []
        yellow_pan_logits = []
        for box, phrase in zip(boxes_filt, pred_phrases):
            if 'yellow pan' in phrase.lower():
                yellow_pan_boxes.append(box)
                yellow_pan_phrases.append(phrase)
                yellow_pan_logits.append(logits)

        crop_img = None

        if len(yellow_pan_boxes) == 0:
            print(f"No 'yellow pan' found in {image_name}, performing center crop")

            # 执行中心裁剪
            img = np.asarray(image_source)

            # crop_rectangle to crop_square
            height, width, _ = img.shape
            # 计算正方形的边长
            L = min(height, width)
            # 计算裁剪区域的左上角坐标
            x = (width - L) // 2
            y = (height - L) // 2
            # 裁剪图像
            crop_img = img[y:y + L, x:x + L]
            #crop_img = img

            # 保存裁剪后的图像
            output_crop_path = os.path.join(output_folder_yellow_pan, f"{image_name.split('.')[0]}.jpg")

            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(output_crop_path, crop_img)

        if len(yellow_pan_boxes) != 0:
            print(f"Found {len(yellow_pan_boxes)} 'yellow pan' in {image_name}")

            # Save yellow pan detection result
            yellow_pan_output_image_path = os.path.join(output_folder_yellow_pan, f"{image_name.split('.')[0]}_bbox.jpg")
            #正常绘制
            #save_output_with_opencv_boxes(frame, yellow_pan_boxes, yellow_pan_phrases, yellow_pan_output_image_path)
            #label超出了图像范围，这种情况下，请将lable和logits绘制到bbox内部的左上角
            #save_output_with_opencv_boxes_(frame,yellow_pan_boxes,yellow_pan_phrases,yellow_pan_output_image_path)


            img = np.asarray(image_source)

            h, w, _ = img.shape
            # boxes = boxes * ([w, h, w, h])

            boxes = boxes_filt * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            boxes = xyxy
            # print('xyxy:',xyxy)
            #print('boxes:', boxes)

            # screen out the box with maximum area
            a = boxes
            e = (a[:, 3] - a[:, 1]) * (a[:, 2] - a[:, 0])
            d = np.argsort(-e, axis=0)  # 按行倒叙排序
            # 出来的是按行，对每一列排序
            #print('max_area_box:,', a[d[0]])
            # crop the maximum area from img (rectangle)
            # crop_rectangle to crop_square (central_point:top_left)
            # crop_rectangle to crop_square (central_point:center point),as the input of yolov5 must be square

            sl = min(int(a[d[0]][3] - a[d[0]][1]), int(a[d[0]][2] - a[d[0]][0]))
            x_center = int((a[d[0]][2] + a[d[0]][0]) / 2)
            y_center = int((a[d[0]][3] + a[d[0]][1]) / 2)

            # 计算裁剪区域的左上角和右下角坐标
            crop_x1 = x_center - int(sl / 2)
            crop_y1 = y_center - int(sl / 2)
            crop_x2 = crop_x1 + sl
            crop_y2 = crop_y1 + sl

            # 裁剪图像
            crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

            #image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            #保存带有标注信息的检测结果
            #annotated_frame = annotate(image_source=image_source, boxes=original_boxes, logits=logits,
                                       #phrases=phrases)


            # 保存裁剪后的图像
            output_crop_path = os.path.join(output_folder_yellow_pan, f"{image_name.split('.')[0]}.jpg")
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            #cv2.imwrite(output_crop_path, crop_img)
#------------------------------------------------------------------------#

        # Filter out only 'wood stick'
        filtered_boxes = []
        filtered_phrases = []
        for box, phrase in zip(boxes_filt, pred_phrases):
            if 'wood stick' in phrase:
                filtered_boxes.append(box)
                filtered_phrases.append(phrase)
        #print('filtered_phrases:',filtered_phrases)
        #print('filtered_boxes:',filtered_boxes)

        predict_number = 0
        mean_confidence_value = 0
        blurriness = 0
        entropy = 0
        niqe = 0
        score = 0
        file_name = ''
        result = open("./simulation_output/score/result_records.txt", 'a')

###------------ 2.wood stick removal by SAM --------------------------###

        # 2.1 no 'wood stick' #
        if len(filtered_boxes) == 0:
            print(f"No 'wood stick' found in {image_name}")
            mask_stirring_tool=None


#-----------------------###3. 执行Yolov5检测 ###----------------------------##

            # (1)轻量化版本

            #start_time_detect_1 = time.time()

            image_name = f"frame_{frame_index:06d}.jpg"
            # 去掉视频文件名，保留目录部分
            video_dir = os.path.dirname(video_source)  # /home/newdrive/simulation/test/test_20241120/video
            # 拼接成完整的路径
            path = os.path.join(video_dir, image_name)
            # keep the original size rather than 640x640
            #print('path:', path)
            file_name_with_extension = os.path.basename(path)
            #print('file_name_with_extension:', file_name_with_extension)
            file_name = os.path.splitext(file_name_with_extension)[0]
            #print('file_name:', file_name)

            # 计算梯度模
            output_image_path = os.path.join(blurriness_folder_path, f'{file_name}_grad.jpg')
            output_blurriness_path = os.path.join(blurriness_folder_path, f'{file_name}_magnitude_plot.png')

            try:
                blurriness, entropy, niqe = compute_gradients_entropy_niqe(crop_img,iqa_model, output_image_path,
                                                                           output_blurriness_path)
                print('Blurriness,entropy,niqe:', blurriness, entropy, niqe)

            except Exception as e:
                print(f'Error processing blurriness_entropy_niqe for {file_name}: {e}')

            pred = detect_on_image (crop_img, model_yolo, stride, names, pt, device="0", imgsz=(640, 640), conf_thres=0.5, iou_thres=0.6)
            #print('pred:',pred)

            # 映射预测框坐标到原图尺寸
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    pred[i][:, :4] = scale_coords(tuple(imgsz), det[:, :4], crop_img.shape).round()
                    #print('pred_:', pred)
            
            
            # Process predictions
            for i, det in enumerate(pred):  # per image
                predict_number = len(det)
                # 提取置信度列（第五列）
                confidences = det[:, 4]
                # 计算平均值
                mean_confidence = torch.mean(confidences)
                mean_confidence_value = mean_confidence.item()
            #start_time_detect_2 = time.time()
            #time_detect = start_time_detect_2 - start_time_detect_1
            #print("Time_detect:", time_detect)




            """
            #(2)非轻量化版本

            #start_detect_time = time.time()
            # Run YOLOv5 detection directly on crop_img
            if crop_img is not None:
                # Convert to BGR if needed
                #if crop_img.shape[2] == 3 and np.all(crop_img[:, :, 0] == crop_img[:, :, 2]):
                    #crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

                # Dataloader
                if webcam:
                    view_img = check_imshow()
                    cudnn.benchmark = True  # set True to speed up constant image size inference
                    dataset = LoadFromImage(source, img_size=imgsz, stride=stride, auto=pt)
                    bs = len(dataset)  # batch_size
                else:
                    #dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
                    dataset = LoadFromImage(crop_img, img_size=imgsz, stride=stride, auto=pt)
                    bs = 1  # batch_size
                vid_path, vid_writer = [None] * bs, [None] * bs
                #result = open("result_records.txt", 'a')


                for path, im, im0s, vid_cap, s in dataset:

                    #video_source = "/home/newdrive/simulation/test/test_20241120/video/Aphid_50.mp4"  # 或者 "your_video.mp4"
                    image_name = f"frame_{frame_index:06d}.jpg"

                    # 去掉视频文件名，保留目录部分
                    video_dir = os.path.dirname(video_source)  # /home/newdrive/simulation/test/test_20241120/video

                    # 拼接成完整的路径
                    path = os.path.join(video_dir, image_name)

                    t1 = time_sync()
                    # keep the original size rather than 640x640
                    #print('path:', path)
                    file_name_with_extension = os.path.basename(path)
                    #print('file_name_with_extension:', file_name_with_extension)
                    file_name = os.path.splitext(file_name_with_extension)[0]
                    #print('file_name:', file_name)

                    # 计算梯度模
                    output_image_path = os.path.join(blurriness_folder_path, f'{file_name}_grad.jpg')
                    output_blurriness_path = os.path.join(blurriness_folder_path, f'{file_name}_magnitude_plot.png')



                    try:
                        blurriness, entropy, niqe = compute_gradients_entropy_niqe(crop_img, iqa_model, output_image_path,
                                                                                   output_blurriness_path)
                        print('Blurriness,entropy,niqe:', blurriness, entropy, niqe)

                    except Exception as e:
                        print(f'Error processing blurriness_entropy_niqe for {file_name}: {e}')

                    im = torch.from_numpy(im).to(device_yolo)
                    im = im.half() if model_yolo.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    t2 = time_sync()
                    dt[0] += t2 - t1

                    # Inference
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if args.visualize else False
                    with torch.no_grad():
                        pred = model_yolo(im, augment=args.augment, visualize=visualize)
                    t3 = time_sync()
                    dt[1] += t3 - t2

                    # NMS
                    # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=args.max_det)

                    # soft-nms
                    pred = soft_nms(pred, args.conf_thres, args.iou_thres, multi_label=True)  # Soft DIoU-NMS
                    #print('pred:', pred)
                    dt[2] += time_sync() - t3


                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        predict_number = len(det)
                        # 提取置信度列（第五列）
                        confidences = det[:, 4]
                        # 计算平均值
                        mean_confidence = torch.mean(confidences)
                        mean_confidence_value = mean_confidence.item()

                        seen += 1
                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        #print('save_path:', save_path)
                        txt_path = str(save_dir / 'labels' / p.stem) + (
                            '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        print('txt_path:', txt_path)
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if args.save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=args.line_thickness, example=str(names))

                        # number print on the image
                        # count_number = "Aphids:{}".format(len(det))
                        # cv2.putText(im0, count_number, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 9)

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                            #print('pred——:', pred)

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if args.save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or args.save_crop or args.view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if args.hide_labels else (
                                        names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
                                    # label = None
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                if args.save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                 BGR=True)



                        # Stream results
                        im0 = annotator.result()
                        if args.view_img:
                            if platform.system() == 'Linux' and p not in windows:
                                windows.append(p)
                                cv2.namedWindow(str(p),
                                                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                            cv2.imshow(str(p), im0)
                            cv2.waitKey(1)  # 1 millisecond

                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                # resize img and print number on the image
                                im0 = cv2.resize(im0, (2000, 2000))
                                count_number = "Aphids:{}".format(len(det))
                                cv2.putText(im0, count_number, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
                                #cv2.imwrite(save_path, im0)

                            else:  # 'video' or 'stream'
                                if vid_path[i] != save_path:  # new video
                                    vid_path[i] = save_path
                                    if isinstance(vid_writer[i], cv2.VideoWriter):
                                        vid_writer[i].release()  # release previous video writer
                                    if vid_cap:  # video
                                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path = str(
                                        Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                                    (w, h))
                                vid_writer[i].write(im0)
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                # Print results
                t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
                LOGGER.info(
                    f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                if args.save_txt or save_img:
                    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if args.save_txt else ''
                    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
                if args.update:
                    strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

                # 确保释放模型内存
                # del model
                # torch.cuda.empty_cache()
                # 记录结束时间
                #end_detect_time = time.time()
                # 计算运行时间
                #elapsed_time = end_detect_time - start_detect_time
                #print(f"time_detect：{elapsed_time:.6f} s")
            """



#----------------------------------------------------------------#


            ## 4. 图像处理部分 ##
            #start_time_analysis_1 = time.time()
            filtered_centroids,aphid_size = analysis(crop_img, image_name, pred, output_folder_color, output_folder_contour,
                                          output_folder_heatmap, output_folder_gradient,
                                          kernel_dilate, kernel_erode, mask_stirring_tool)
            #print('filtered_centroids:', filtered_centroids)
            #start_time_analysis_2 = time.time()
            #time_analysis = start_time_analysis_2 - start_time_analysis_1
            #print("time_analysis:", time_analysis)

            del pred

            ## 5. 密度分析 ##
            if len(filtered_centroids) > 1:

                img_height, img_width = crop_img.shape[:2]
                # Convert filtered_centroids to numpy array
                centroids = np.array(filtered_centroids)

                # Initialize analyzer
                analyzer = AphidAnalyzer(aphid_size, centroids, (img_width, img_height),
                                         os.path.basename(input_image_path).split('.')[0],labels_dir)

                #print(os.path.basename(input_image_path))
                print(f"\n=== Analyzing image: {os.path.basename(input_image_path)} ===")
                print(f"Automatically set parameters: {analyzer.params}")

                # Run full analysis
                #start_time_distribution_1 = time.time()
                #analyzer.analyze_distribution()
                score = analyzer.analyze_distribution()
                print(f"Returned clustering score: {score:.2f}")
                #start_time_distribution_2 = time.time()
                #time_distribution = start_time_distribution_2 - start_time_distribution_1
                #print("time_distribution_analysis:", time_distribution)




            #result.write(
                #file_name + '.jpg' + ',' + str(mean_confidence_value) + ',' + str(predict_number) + ',' + str(
                    #blurriness) + ',' + str(entropy) + ',' + str(niqe) + ',' + str(score)+'\n')
            #result.close()



            # 记录结束时间
            #end_time = time.time()
            # 计算运行时间
            #elapsed_time = end_time - start_time
            #print(f"Running time：{elapsed_time:.6f} s")
            #time_records.append(elapsed_time)
            #print("time_records:", time_records)

            del crop_img
            del analyzer, centroids




        # 2.2 yes 'wood stick'
        if len(filtered_boxes) != 0:
            filtered_boxes = torch.stack(filtered_boxes)
            size = image_pil.size
            H, W = size[1], size[0]

            for i in range(filtered_boxes.size(0)):
                filtered_boxes[i] = filtered_boxes[i] * torch.Tensor([W, H, W, H])
                filtered_boxes[i][:2] -= filtered_boxes[i][2:] / 2
                filtered_boxes[i][2:] += filtered_boxes[i][:2]

            # 将原始图像中使用dino检测到的'wood stick' box 转换为 crop_img 中的坐标
            # 只提取yellow_pan区域的wood stick部分
            adjusted_boxes = []
            for box in filtered_boxes:
                x1, y1, x2, y2 = box.tolist()

                # 判断 box 是否与 crop_img 有交集（即部分或全部在裁剪区域内）
                if x2 <= crop_x1 or x1 >= crop_x2 or y2 <= crop_y1 or y1 >= crop_y2:
                    continue  # 完全在裁剪外，跳过

                # 裁剪为 crop_img 内的坐标
                new_x1 = max(x1 - crop_x1, 0)
                new_y1 = max(y1 - crop_y1, 0)
                new_x2 = min(x2 - crop_x1, crop_img.shape[1])
                new_y2 = min(y2 - crop_y1, crop_img.shape[0])

                adjusted_boxes.append([new_x1, new_y1, new_x2, new_y2])

            if not adjusted_boxes:
                print(f"[WARN] No valid 'wood stick' inside crop area in {image_name}")
                continue

            # 转为 tensor
            filtered_boxes_crop = torch.tensor(adjusted_boxes, dtype=torch.float)

            # Prepare for SAM prediction with crop_img
            image_cv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            #result = open("result_records.txt", 'a')

    #-----------------------  ### 3. 执行Yolov5检测 ###----------------------------#

            # (1)轻量化版本

            #start_time_detect_1 = time.time()
            image_name = f"frame_{frame_index:06d}.jpg"
            # 去掉视频文件名，保留目录部分
            video_dir = os.path.dirname(video_source)  # /home/newdrive/simulation/test/test_20241120/video
            # 拼接成完整的路径
            path = os.path.join(video_dir, image_name)
            # keep the original size rather than 640x640
            #print('path:', path)
            file_name_with_extension = os.path.basename(path)
            #print('file_name_with_extension:', file_name_with_extension)
            file_name = os.path.splitext(file_name_with_extension)[0]
            #print('file_name:', file_name)

            # 计算梯度模
            output_image_path = os.path.join(blurriness_folder_path, f'{file_name}_grad.jpg')
            output_blurriness_path = os.path.join(blurriness_folder_path, f'{file_name}_magnitude_plot.png')

            try:
                blurriness, entropy, niqe = compute_gradients_entropy_niqe(crop_img, iqa_model, output_image_path,
                                                                           output_blurriness_path)
                print('Blurriness,entropy,niqe:', blurriness, entropy, niqe)

            except Exception as e:
                print(f'Error processing blurriness_entropy_niqe for {file_name}: {e}')

            pred = detect_on_image(crop_img, model_yolo, stride, names, pt, device="0", imgsz=(640, 640), conf_thres=0.5, iou_thres=0.6)
            #print('pred:', pred)
            # 映射预测框坐标到原图尺寸
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    pred[i][:, :4] = scale_coords(tuple(imgsz), det[:, :4], crop_img.shape).round()
                    #print('pred_:', pred)



            # Process predictions
            for i, det in enumerate(pred):  # per image
                predict_number = len(det)
                # 提取置信度列（第五列）
                confidences = det[:, 4]
                # 计算平均值
                mean_confidence = torch.mean(confidences)
                mean_confidence_value = mean_confidence.item()
            #start_time_detect_2 = time.time()
            #time_detect = start_time_detect_2 - start_time_detect_1
            #print("Time_detect:", time_detect)



            """
            # (2)非轻量化版本
            #start_detect_time = time.time()
            # Run YOLOv5 detection directly on crop_img
            if crop_img is not None:
                # Convert to BGR if needed
                # if crop_img.shape[2] == 3 and np.all(crop_img[:, :, 0] == crop_img[:, :, 2]):
                # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)

                # Dataloader
                if webcam:
                    view_img = check_imshow()
                    cudnn.benchmark = True  # set True to speed up constant image size inference
                    dataset = LoadFromImage(source, img_size=imgsz, stride=stride, auto=pt)
                    bs = len(dataset)  # batch_size
                else:
                    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
                    dataset = LoadFromImage(crop_img, img_size=imgsz, stride=stride, auto=pt)
                    bs = 1  # batch_size
                vid_path, vid_writer = [None] * bs, [None] * bs
                #result = open("result_records.txt", 'a')

                for path, im, im0s, vid_cap, s in dataset:

                    image_name = f"frame_{frame_index:06d}.jpg"

                    # 去掉视频文件名，保留目录部分
                    video_dir = os.path.dirname(video_source)  # /home/newdrive/simulation/test/test_20241120/video

                    # 拼接成完整的路径
                    path = os.path.join(video_dir, image_name)

                    t1 = time_sync()
                    # keep the original size rather than 640x640
                    #print('path:', path)
                    file_name_with_extension = os.path.basename(path)
                    #print('file_name_with_extension:', file_name_with_extension)
                    file_name = os.path.splitext(file_name_with_extension)[0]
                    #print('file_name:', file_name)

                    # 计算梯度模
                    output_image_path = os.path.join(blurriness_folder_path, f'{file_name}_grad.jpg')
                    output_blurriness_path = os.path.join(blurriness_folder_path, f'{file_name}_magnitude_plot.png')


                    try:
                        blurriness, entropy, niqe = compute_gradients_entropy_niqe(crop_img, iqa_model, output_image_path,
                                                                                   output_blurriness_path)
                        print('Blurriness,entropy,niqe:', blurriness, entropy, niqe)

                    except Exception as e:
                        print(f'Error processing blurriness_entropy_niqe for {file_name}: {e}')

                    im = torch.from_numpy(im).to(device_yolo)
                    im = im.half() if model_yolo.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    t2 = time_sync()
                    dt[0] += t2 - t1

                    # Inference
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if args.visualize else False
                    with torch.no_grad():
                        pred = model_yolo(im, augment=args.augment, visualize=visualize)
                    t3 = time_sync()
                    dt[1] += t3 - t2

                    # NMS
                    # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=args.max_det)

                    # soft-nms
                    pred = soft_nms(pred, args.conf_thres, args.iou_thres, multi_label=True)  # Soft DIoU-NMS
                    #print('pred:', pred)
                    dt[2] += time_sync() - t3

                    # Process predictions
                    for i, det in enumerate(pred):  # per image
                        predict_number = len(det)
                        # 提取置信度列（第五列）
                        confidences = det[:, 4]
                        # 计算平均值
                        mean_confidence = torch.mean(confidences)
                        mean_confidence_value = mean_confidence.item()
                        #print('mean_confidence_value,len(det):', mean_confidence_value, len(det))
                        # print('len(det):', len(det))

                        seen += 1
                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f'{i}: '
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # im.jpg
                        #print('save_path:', save_path)
                        txt_path = str(save_dir / 'labels' / p.stem) + (
                            '' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                        print('txt_path:', txt_path)
                        s += '%gx%g ' % im.shape[2:]  # print string
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if args.save_crop else im0  # for save_crop
                        annotator = Annotator(im0, line_width=args.line_thickness, example=str(names))

                        # number print on the image
                        # count_number = "Aphids:{}".format(len(det))
                        # cv2.putText(im0, count_number, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 9)

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if args.save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
                                    with open(f'{txt_path}.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                if save_img or args.save_crop or args.view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if args.hide_labels else (
                                        names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
                                    # label = None
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                                if args.save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                 BGR=True)

                        #result.write(file_name + '.jpg' + ',' + str(mean_confidence_value) + ',' + str(len(det)) + ',' + str(
                                #blurriness) + ',' + str(entropy) + ',' + str(niqe) + '\n')
                        #result.close()

                        # Stream results
                        im0 = annotator.result()
                        if args.view_img:
                            if platform.system() == 'Linux' and p not in windows:
                                windows.append(p)
                                cv2.namedWindow(str(p),
                                                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                            cv2.imshow(str(p), im0)
                            cv2.waitKey(1)  # 1 millisecond

                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                # resize img and print number on the image
                                im0 = cv2.resize(im0, (2000, 2000))
                                count_number = "Aphids:{}".format(len(det))
                                cv2.putText(im0, count_number, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 10)
                                #cv2.imwrite(save_path, im0)
                            else:  # 'video' or 'stream'
                                if vid_path[i] != save_path:  # new video
                                    vid_path[i] = save_path
                                    if isinstance(vid_writer[i], cv2.VideoWriter):
                                        vid_writer[i].release()  # release previous video writer
                                    if vid_cap:  # video
                                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path = str(
                                        Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                                                    (w, h))
                                vid_writer[i].write(im0)
                # Print time (inference-only)
                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                # Print results
                t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
                LOGGER.info(
                    f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
                if args.save_txt or save_img:
                    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if args.save_txt else ''
                    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
                if args.update:
                    strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

                # 确保释放模型内存
                # del model
                # torch.cuda.empty_cache()
                # 记录结束时间
                #end_detect_time = time.time()
                # 计算运行时间
                #elapsed_time = end_detect_time - start_detect_time
                #print(f"time_detect：{elapsed_time:.6f} s")
            """



            #----------------------------------------------------------------#

            # 确保释放YOLOv5占用的GPU资源
            #time.sleep(1)  # 给YOLOv5时间释放资源
            torch.cuda.empty_cache()
            #start_time_sam_1 = time.time()
            predictor.set_image(image_cv)

            # 将 box 映射为 SAM 所需格式
            transformed_boxes = predictor.transform.apply_boxes_torch(filtered_boxes_crop, image_cv.shape[:2]).to(device)

            # Predict masks using SAM
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            #start_time_sam_2 = time.time()
            #time_sam = start_time_sam_2 - start_time_sam_1
            #print("time_sam:", time_sam)



            # Clip masks to their respective boxes
            # 调整之后的box的区域内做sam分割，但是分割的mask怎么会超出adjusted box的范围? 什么原因？
            # SAM uses boxes as positional hints rather than strict boundaries, allowing it to find connected regions that may extend outside the box.
            # 在 SAM 预测后立即应用 box 约束，确保 mask 不超出检测框范围，
            # 其实没必要，按照SAM的设计原理即可,也合理
            """
            for i, (mask, box) in enumerate(zip(masks, filtered_boxes_crop)):
                x1, y1, x2, y2 = map(int, box.tolist())
                box_mask = torch.zeros_like(mask)
                box_mask[:, y1:y2, x1:x2] = 1
                masks[i] = mask * box_mask
            """

            #save_mask_data(args.output_dir, masks, filtered_boxes_crop, filtered_phrases, image_name)

            # 将 masks 转换为 numpy 格式
            masks_np = masks.cpu().numpy()

            # masks 形状假设是 (1, H, W)，去掉 batch 维度
            masks_np = masks_np.squeeze()

            # 转换为 uint8 类型用于轮廓提取
            masks_uint8 = (masks_np * 255).astype(np.uint8)

            # 提取所有轮廓
            contours, _ = cv2.findContours(masks_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 初始化最大轮廓和面积
            largest_contour = None
            max_area = 0

            # 遍历所有轮廓，找到面积最大的轮廓
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > max_area:
                    largest_contour = contour
                    max_area = area

            # 创建一个空白的 mask 图像，用于绘制最大轮廓
            largest_mask = np.zeros_like(masks_uint8)

            # 如果找到了最大的轮廓，则绘制该轮廓
            if largest_contour is not None:
                cv2.drawContours(largest_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

            # 将 largest_mask 转换回浮点类型并归一化为二值化 (0 和 1)
            largest_mask = (largest_mask / 255).astype(np.float32)

            # Save wood stick bbox and yellow_mask
            output_image_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_output.jpg")
            #img, mask_stirring_tool = save_output_with_opencv(image_cv, largest_mask, filtered_boxes_crop, filtered_phrases, output_image_path)

            # Save mask data
            output_image_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_largest_mask.jpg")
            #cv2.imwrite(output_image_path, largest_mask * 255)

            # 在SAM预测后（save_mask_data之前）
            del masks, transformed_boxes, masks_np, masks_uint8



            ## 4. 图像处理 ##
            filtered_centroids, aphid_size = analysis(crop_img, image_name, pred, output_folder_color,
                                                      output_folder_contour,
                                                      output_folder_heatmap, output_folder_gradient,
                                                      kernel_dilate, kernel_erode, mask_stirring_tool)

            #print('filtered_centroids:',filtered_centroids)

            del pred

            ## 5. 密度分析 ##
            if len(filtered_centroids) > 1:

                # Load image to get dimensions
                img_height, img_width = crop_img.shape[:2]

                # Convert filtered_centroids to numpy array
                centroids = np.array(filtered_centroids)

                # Initialize analyzer
                analyzer = AphidAnalyzer(aphid_size, centroids, (img_width, img_height),
                                         os.path.basename(input_image_path).split('.')[0], labels_dir)
                print(f"\n=== Analyzing image: {os.path.basename(input_image_path)} ===")
                print(f"Automatically set parameters: {analyzer.params}")

                # Run full analysis
                #analyzer.analyze_distribution()
                score = analyzer.analyze_distribution()
                print(f"Returned clustering score: {score:.2f}")

            # 在analyzer.analyze_distribution()之后
            del analyzer, centroids
            del crop_img, largest_mask

#####################perception<->speed###############################################
        # ==== 存储本帧指标 ====
        current_metrics = {
            "PDU": score,
            "MeanConfidence": mean_confidence_value,
            "Blurriness": blurriness,
            "IC": entropy,
            "IQA": niqe,
            "Detections": predict_number
        }

        history.append(current_metrics)
        # 保证历史至少有 k 条记录后才滑动计算
        if len(history) >= k:
            # 滑动窗口：取当前时间点向前 k 秒的连续指标
            sliding_window = history[-k:]
            delta_total = compute_delta_total(sliding_window, weights,negatives)
            change_total.append(delta_total)

            print(f"\n[Time {frame_index}s] Δ_total = {delta_total:.4f}")

            if abs(delta_total) < stop_threshold:
                speed = 0.0
                print(f" → Change less than {stop_threshold}，stop stirring")
            else:
                # speed += delta_total * adjust_factor
                # speed = max(min_speed, min(max_speed, speed))
                speed += np.tanh(delta_total) * adjust_factor
                print(f" → update speed: speed = {speed:.3f}")

            speed_records.append(speed)

        else:
            print(f"\n[Time {frame_index}s] Waiting for collecting {k} group data ...")



        print('change_total：', change_total)
        print('speed_records：', speed_records)


#######################################################################

        frame_index += 1
        # 记录结束时间
        end_time = time.time()
        # 计算运行时间
        elapsed_time = end_time - start_time
        print(f"Running time：{elapsed_time:.6f} s")
        time_records.append(elapsed_time)
        print("time_records:", time_records)

        result.write(file_name + '.jpg' + ',' + str(mean_confidence_value) + ',' + str(predict_number) + ',' + str(
            blurriness) + ',' + str(entropy) + ',' + str(niqe) + ',' + str(score) + '\n')
        result.close()

        #del intermediate_results
        # 在每帧处理结束的位置（while循环末尾）添加以下代码：
        del boxes_filt, pred_phrases, logits  # GroundingDINO输出
        del image_pil, image  # 图像数据（确保后续不再使用）

        # 显式清空CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()





if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything for Directory", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to GroundingDINO checkpoint file")
    parser.add_argument("--sam_version", type=str, default="vit_b", help="SAM ViT version: vit_b / vit_l / vit_h")
    #parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to SAM checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to SAM-HQ checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="use SAM-HQ for prediction")
    parser.add_argument("--input_dir", type=str, required=True, help="input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory to save results")
    #parser.add_argument("--text_prompt", type=str, default="yellow panbase. yellow panedge. wood stick.", help="text prompt for grounding")
    parser.add_argument("--text_prompt", type=str, default="yellow pan. wood stick.",
                        help="text prompt for grounding")
    #parser.add_argument("--text_prompt", type=str, default="the_inner_circumference. wood stick.",
                        #help="text prompt for grounding")

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold for filtering")

    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold for filtering")
    parser.add_argument("--device", type=str, default="cuda", help="device to run the model (cpu or cuda)")


    #Yolov5
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device_yolo', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='counting_confidence', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--xml', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')

    args = parser.parse_args()

    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    print_args(vars(args))


    #args = parser.parse_args()

    # 定义 YOLO 标签文件目录
    labels_dir = "./runs/detect/counting_confidence/labels"

    process_directory(args.input_dir, args.output_dir, args)
    #process_directory(**vars(args))























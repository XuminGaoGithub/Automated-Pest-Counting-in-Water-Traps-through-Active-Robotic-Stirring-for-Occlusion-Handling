'''
#parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to SAM-HQ checkpoint file")
#--sam_hq_checkpoint: python counting_confidence_grounded_sam.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /home/xumin/Open-GroundingDino/weights/groundingdino_swint_ogc.pth --sam_hq_checkpoint sam_hq_vit_b.pth --use_sam_hq  --input_dir /home/newdrive/summerschool/Grounded-Segment-Anything/counting_confidence/src --output_dir "./simulation_output/image_processing" --box_threshold 0.5 --text_threshold 0.25 --device "cuda"

#parser.add_argument("--sam_checkpoint", type=str, required=True, help="path to SAM checkpoint file")
#--sam_checkpoint: python counting_confidence_grounded_sam.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /home/xumin/Open-GroundingDino/weights/groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_b_01ec64.pth  --input_dir /home/newdrive/summerschool/Grounded-Segment-Anything/counting_confidence/src --output_dir "./simulation_output/image_processing" --box_threshold 0.5 --text_threshold 0.25 --device "cuda"


#input_image no resize
#mask = cv2.erode(mask, kernel_erode, iterations=2)

#input_image resize
#mask = cv2.erode(mask, kernel_erode, iterations=1)

'''



import argparse
import os
import sys
import numpy as np
import json
import torch
from PIL import Image

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

warnings.simplefilter(action='ignore', category=FutureWarning)


# 创建结果目录
os.makedirs("./simulation_output/analysis", exist_ok=True)
os.makedirs("./simulation_output/score", exist_ok=True)


import os
import time



def is_file_fully_written(filepath, wait_time=0.2, retries=5):
    """检查文件是否已经完全写入：短时间内大小不变视为写入完成"""
    last_size = -1
    for _ in range(retries):
        try:
            current_size = os.path.getsize(filepath)
            if current_size == last_size:
                return True
            last_size = current_size
            time.sleep(wait_time)
        except OSError:
            time.sleep(wait_time)
    return False


def wait_for_complete_file(file_path, wait_interval=0.05, stable_time=0.5, timeout=1000):
    """
    等待文件完全写入，直到其大小在 stable_time 内不再变化，或超时。
    """
    start_time = time.time()
    last_size = -1
    stable_start = None

    while True:
        if os.path.exists(file_path):
            current_size = os.path.getsize(file_path)
            if current_size == last_size:
                if stable_start is None:
                    stable_start = time.time()
                elif time.time() - stable_start >= stable_time:
                    # 文件大小在稳定时间内没有变化 -> 假定写入完成
                    break
            else:
                # 文件大小变化了，重置稳定检测
                stable_start = None
                last_size = current_size
        else:
            # 文件尚未生成
            stable_start = None
            last_size = -1

        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timed out waiting for file to stabilize: {file_path}")

        time.sleep(wait_interval)



# ================== 参数自动设置模块 ==================
def calculate_aphid_size(image_name, labels_dir, img_width, img_height):
    """
    计算蚜虫的平均大小（基于检测框的宽度和高度）
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
        print(f"Label file not found: {label_path}")
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


def auto_set_parameters(centroids, img_width, img_height, image_name, labels_dir):
    """自动设置所有聚类相关参数"""
    # 计算蚜虫平均大小
    #print('image_name:', image_name)
    #print('labels_dir:', labels_dir)
    #print('w:', img_width)
    #print('h:', img_height)
    aphid_size = calculate_aphid_size(image_name, labels_dir, img_width, img_height)
    threshold = 1.5 * aphid_size  # 动态阈值
    print('avg_aphid_size:', aphid_size)

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


# ================== 核心分析模块 ==================
class AphidAnalyzer:
    def __init__(self, centroids, img_size, image_name, labels_dir):
        self.centroids = centroids
        self.img_width, self.img_height = img_size
        self.image_name = image_name
        self.labels_dir = labels_dir
        self.params = auto_set_parameters(centroids, self.img_width, self.img_height, self.image_name, self.labels_dir)

    def analyze_distribution(self):
        """执行完整分析流程"""
        self._basic_analysis()
        if not self.params['no_clustering']:
            self._run_all_clustering()
            self._visualize_all_results()

            # 将评分结果保存到 result.txt 文件
            with open("./simulation_output/score/result.txt", "a") as f:
                for method, score in self.cluster_scores.items():
                    f.write(f"{self.image_name + '.jpg'},{method},{score:.2f}\n")
        else:
            # 即使没有聚类，也将结果写入文件，score 设置为 0
            with open("./simulation_output/score/result.txt", "a") as f:
                f.write(f"{self.image_name + '.jpg'},'Adaptive DBSCAN',0.00\n")
            print("No clustering detected, skipping clustering step.")


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

            print('Avg_distance of' + 'Clustering' + ' ' + str(label) + ':', avg_distance)

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
    model = model.to(device)
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

    '''
    json_data = [{'value': value, 'label': 'background'}]
    for label, box in zip(label_list, box_list):
        value += 1

        if '(' in label and ')' in label:
            name, logit = label.split('(')
            logit = logit[:-1]
        else:
            name = label
            logit = None

        box_cpu = box.cpu()

        json_data.append({
            'value': value,
            'label': name.strip(),
            'logit': float(logit) if logit is not None else None,
            'box': box_cpu.numpy().tolist(),
        })

    json_filename = f"{os.path.splitext(image_name)[0]}_mask.json"
    with open(os.path.join(output_dir, json_filename), 'w') as f:
        json.dump(json_data, f)
    '''


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
    cv2.imwrite(output_image_path.replace('.jpg', '_bbox.jpg'), copy_image_bgr_1)

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
    cv2.imwrite(output_image_path.replace('.jpg', '_yellow_mask.jpg'), copy_image_bgr_2)

    return copy_image_bgr_2,mask_indices


def analysis(img, image_name, output_folder_color, output_folder_contour,output_folder_heatmap, output_folder_gradient,
             kernel_dilate, kernel_erode,grid_size,mask_stirring_tool):

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
            ##cv2.imwrite(gradient_output_path, grad_magnitude_display)

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
            ##cv2.imwrite(gradient_output_path, grad_magnitude_display)
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
        cv2.imwrite(output_contour_mask_path, inverted_mask)

        # 颜色分割结果：去除黄色背景
        result_color = cv2.bitwise_and(img, img, mask=inverted_mask)
        # 保存 original_color_segmented 图像
        original_color_segmented_path = os.path.join(output_folder_color, f'original_color_segmented_{image_name}')
        cv2.imwrite(original_color_segmented_path, result_color)


        # 膨胀和腐蚀处理
        mask=inverted_mask
        #mask = cv2.dilate(inverted_mask, kernel_dilate, iterations=2)
        #mask = cv2.erode(mask, kernel_erode, iterations=2)
        mask = cv2.erode(mask, kernel_erode, iterations=1)

        #inverted_mask = cv2.bitwise_not(mask)

        # 保存原始 mask 图像
        original_mask_path = os.path.join(output_folder_contour, f'original_morphology_mask_{image_name}')
        cv2.imwrite(original_mask_path, mask)



        # 形态学处理，闭运算（膨胀+腐蚀）连接靠近的物体
        # morph_mask = cv2.morphologyEx(inverted_mask, cv2.MORPH_CLOSE, kernel_dilate)
        morph_mask = mask
        # 查找轮廓并计算质心
        contours, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        #result_contour = img.copy()
        mask_with_contours = np.stack([morph_mask] * 3, axis=-1)  # 将单通道mask扩展为3通道以绘制轮廓和质心

        # 绘制轮廓到mask图像和原图
        #cv2.drawContours(result_contour, contours, -1, (0, 0, 255), 2)
        cv2.drawContours(mask_with_contours, contours, -1, (0, 0, 255), 2)  # 绿色轮廓

        # 保存轮廓和质心的原图和mask图
        # output_contour_path = os.path.join(output_folder_contour, f'original_contour_{image_name}')
        # #cv2.imwrite(output_contour_path, result_contour)

        output_contour_mask_path = os.path.join(output_folder_contour, f'original_contour_mask_before_filter_noise_{image_name}')
        cv2.imwrite(output_contour_mask_path, mask_with_contours)



        # 计算蚜虫的平均大小
        mask_with_contours_filter_noise = np.stack([morph_mask] * 3, axis=-1)  #
        image_name_without_ext = os.path.splitext(image_name)[0]  # 去掉 .jpg 后缀
        #print('w, h:',w, h)
        aphid_size = calculate_aphid_size(image_name_without_ext, labels_dir, w, h)
        #print('aphid_size:',aphid_size)

        min_contour_area = (aphid_size * aphid_size) / 10  # 最小轮廓面积为 aphid_size 的 1/10
        #min_contour_area = (aphid_size * aphid_size) / 20  # 最小轮廓面积为 aphid_size 的 1/10

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
                cv2.drawContours(mask_with_contours_filter_noise, [contour], -1, 0, thickness=cv2.FILLED)  # 填充为黑色
                continue

            # 保留符合条件的轮廓
            filtered_contours_1.append(contour)

        # 绘制轮廓到mask图像和原图
        #cv2.drawContours(result_contour, filtered_contours_1, -1, (0, 0, 255), 2)
        cv2.drawContours(mask_with_contours_filter_noise, filtered_contours_1, -1, (0, 0, 255), 2)  # 绿色轮廓

        # 保存轮廓和质心的原图和mask图
        #output_contour_path = os.path.join(output_folder_contour, f'original_contour_{image_name}')
        ##cv2.imwrite(output_contour_path, result_contour)

        output_contour_mask_path = os.path.join(output_folder_contour, f'original_contour_mask_after_filter_noise_{image_name}')
        cv2.imwrite(output_contour_mask_path, mask_with_contours_filter_noise)

        for contour in filtered_contours_1:
            # 计算质心
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                # 在原图上画质心（蓝色）
                #cv2.circle(result_contour, (cx, cy), 5, (0, 0, 255), -1)
                # 在mask图上画质心（蓝色）
                cv2.circle(mask_with_contours_filter_noise, (cx, cy), 5, (0, 0, 255), -1)

        # 绘制轮廓到mask图像和原图
        #cv2.drawContours(result_contour, filtered_contours_1, -1, (0, 0, 255), 2)
        cv2.drawContours(mask_with_contours_filter_noise, filtered_contours_1, -1, (0, 0, 255), 2)  # 绿色轮廓

        # 保存绘制了轮廓和质心的原图和mask图
        #output_contour_path = os.path.join(output_folder_contour, f'original_contour_with_centroids_{image_name}')
        ##cv2.imwrite(output_contour_path, result_contour)

        #output_contour_mask_path = os.path.join(output_folder_contour, f'original_contour_with_centroids_mask_{image_name}')
        ##cv2.imwrite(output_contour_mask_path, mask_with_contours_filter_noise)

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
        cv2.drawContours(filtered_mask, filtered_contours, -1, (255), thickness=cv2.FILLED)


        #cv2.drawContours(filtered_mask, filtered_contours, -1, (0, 0, 255), 2)
        #filtered_mask = np.stack([mask] * 3, axis=-1)
        #cv2.drawContours(filtered_mask, filtered_contours, -1, (0, 0, 255), 2)

        # 保存过滤后的mask
        #filtered_mask_path = os.path.join(output_folder_color, f'after_filter_corner_contour_mask_{image_name}')
        ##cv2.imwrite(filtered_mask_path, filtered_mask)
        # 保存 after_filtered_contour_mask 图像，带质心
        after_filtered_contour_mask_with_centroids = np.stack([filtered_mask] * 3, axis=-1)
        cv2.drawContours(after_filtered_contour_mask_with_centroids, filtered_contours, -1, (0, 0, 255), 2)
        after_filtered_contour_mask_path = os.path.join(output_folder_contour,
                                                        f'after_filter_corner_contour_mask_{image_name}')
        cv2.imwrite(after_filtered_contour_mask_path, after_filtered_contour_mask_with_centroids)

        # 保存过滤后的颜色分割结果
        #filtered_color_segmented = cv2.bitwise_and(img, img, mask=filtered_mask)
        #filtered_output_color_path = os.path.join(output_folder_color, f'after_filter_corner_color_segmented_{image_name}')
        ##cv2.imwrite(filtered_output_color_path, filtered_color_segmented)

        # 保存过滤后的轮廓图像
        #filtered_result_contour = img.copy()
        #cv2.drawContours(filtered_result_contour, filtered_contours, -1, (0, 0, 255), 2)
        #filtered_output_contour_path = os.path.join(output_folder_contour,
                                                    #f'after_filter_corner_contour_with_centroids_{image_name}')
        ##cv2.imwrite(filtered_output_contour_path, filtered_result_contour)

        # 保存 after_filtered_contour_mask 图像，带质心
        after_filtered_contour_mask_with_centroids = np.stack([filtered_mask] * 3, axis=-1)
        cv2.drawContours(after_filtered_contour_mask_with_centroids, filtered_contours, -1, (0, 0, 255), 2)
        for cx, cy in filtered_centroids:
            cv2.circle(after_filtered_contour_mask_with_centroids, (cx, cy), 5, (0, 0, 255), -1)
        after_filtered_contour_mask_path = os.path.join(output_folder_contour,
                                                        f'after_filter_corner_contour_mask_with_centroids_{image_name}')
        cv2.imwrite(after_filtered_contour_mask_path, after_filtered_contour_mask_with_centroids)

        filtered_mask = np.zeros_like(mask)
        only_centroids_mask = np.stack([filtered_mask] * 3, axis=-1)
        #cv2.drawContours(after_filtered_contour_mask_with_centroids, filtered_contours, -1, (0, 0, 255), 2)
        for cx, cy in filtered_centroids:
            cv2.circle(only_centroids_mask, (cx, cy), 5, (0, 0, 255), -1)
        after_filtered_contour_mask_path = os.path.join(output_folder_contour,
                                                        f'only_centroids_{image_name}')
        cv2.imwrite(after_filtered_contour_mask_path, only_centroids_mask)

        #print('filtered_centroids:',filtered_centroids)
    return filtered_centroids



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

    # 设置结构元素大小（膨胀和腐蚀的核）
    kernel_dilate = np.ones((7, 7), np.uint8)
    kernel_erode = np.ones((3, 3), np.uint8)

    # 定义网格大小
    grid_size = 100

    device = torch.device(args.device)

    # Load Grounding DINO model
    model = load_model(args.config, args.grounded_checkpoint, device=device)

    # Initialize SAM
    if args.use_sam_hq:
        predictor = SamPredictor(
            sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(device))

    # Initialize SAM
    #if args.use_sam_hq:
        #sam = sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint).to(device)
    #else:
        #sam = sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(device)
    #predictor = SamPredictor(sam)


    processed_images = set()

    # 改为0为摄像头，或者写入视频路径例如 "video.mp4"
    video_source = "/home/newdrive/simulation/test/test_20241120/video/Aphid_50.mp4"  # 或者 "your_video.mp4"

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


    #读间隔2s取每一帧
    """
    while True:
        start_time = time.time()

        # 计算下一帧的时间点，单位毫秒
        ms_pos = frame_index * interval_sec * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, ms_pos)

        ret, frame = cap.read()

        if not ret:
            print("[INFO] 视频读取结束或失败，等待2秒后重启读取...")
            #time.sleep(2)
            cap.release()
            cap = cv2.VideoCapture(video_source)
            frame_index = 0
            continue
    """



    while True:
        start_time = time.time()
        # 计算下一帧的时间点（单位毫秒）
        ms_pos = frame_index * interval_sec * 1000
        cap.set(cv2.CAP_PROP_POS_MSEC, ms_pos)
        ret, frame = cap.read()
        if not ret:
            print("[INFO] 视频读取结束或失败。")
            break  # 不再重启，直接退出循环


        # 2. 保存图像帧
        img_name = f"frame_{frame_index:06d}.jpg"
        img_path = os.path.join(input_dir, img_name)

        #将图像 resize to save GPU memory#
        #height, width = frame.shape[:2]
        #frame = cv2.resize(frame, (width // 2, height // 2))


        # 将图像 resize to save GPU memory#
        # 获取原始尺寸
        height, width = frame.shape[:2]
        # 计算缩放比例（将最短边缩放到 640）
        short_side = min(height, width)
        scale = 640.0 / short_side
        # 计算新的尺寸
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        # 缩放图像
        frame = cv2.resize(frame, (new_width, new_height))


        # 保存图像
        cv2.imwrite(img_path, frame)



        frame_index += 1

        # 3. 等待写入完成
        if not is_file_fully_written(img_path):
            print(f"[WARN] 文件未写入完成: {img_name}，跳过")
            continue

        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
        new_images = [f for f in image_files if f not in processed_images]

        if not new_images:
            time.sleep(0.01)
            print(f"Waiting for new images in {input_dir}...")
            continue

        input_image_path =str('images/')
        #source = str(source)
        for image_name in new_images:
            input_image_path = os.path.join(input_dir, image_name)
            if is_file_fully_written(input_image_path):
                processed_images.add(image_name)
                print(f"[INFO] New image detected: {image_name}, running detection...")
                # 在这里执行你的读取和处理操作，比如：
                # image = cv2.imread(source)
            else:
                print(f"[WARN] File not ready yet: {image_name}, will check again later.")


            print('input_image_path:',input_image_path)
            print(f"[INFO] New image analysis: {image_name}, running analysis...")


        # Load image
        image_pil, image = load_image(input_image_path)
        image_source = np.asarray(image_pil)


        # Run Grounding DINO model
        boxes_filt, pred_phrases, logits, tokenized = get_grounding_output(
            model, image, args.text_prompt, args.box_threshold, args.text_threshold, device=device
        )



        # 检查是否有 'yellow pan' 在检测结果中
        yellow_pan_boxes = []
        for box, phrase in zip(boxes_filt, pred_phrases):
            if 'yellow pan' in phrase.lower():
                yellow_pan_boxes.append(box)

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
            #crop_img = cv2.resize(crop_img, (640, 640)) #save GPU memory->SAM

            #crop_img = img

            # 保存裁剪后的图像
            output_crop_path = os.path.join(output_folder_yellow_pan, f"{image_name.split('.')[0]}.jpg")
            cv2.imwrite(output_crop_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))



        if len(yellow_pan_boxes) != 0:
            print(f"Found {len(yellow_pan_boxes)} 'yellow pan' in {image_name}")

            #image_source = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
            img = np.asarray(image_source)

            h, w, _ = img.shape
            # boxes = boxes * ([w, h, w, h])

            boxes = boxes_filt * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            boxes = xyxy
            # print('xyxy:',xyxy)
            #print('boxes:', boxes)

            # screen out the box with maximum area
            # a = np.array(boxes)
            a = boxes

            # print(a[:, 3] - a[:, 1])
            # print(a[:, 2] - a[:, 0])
            e = (a[:, 3] - a[:, 1]) * (a[:, 2] - a[:, 0])
            # print('areas,:',e)
            d = np.argsort(-e, axis=0)  # 按行倒叙排序
            # 出来的是按行，对每一列排序
            #print('max_area_box:,', a[d[0]])
            # crop the maximum area from img (rectangle)
            #crop_img = img[int(a[d[0]][1]):int(a[d[0]][3]), int(a[d[0]][0]):int(a[d[0]][2])]

            # crop_rectangle to crop_square (central_point:top_left)
            # sl=min(int(a[d[0]][3]-a[d[0]][1]),int(a[d[0]][2]-a[d[0]][0]))
            # crop_img = img[int(a[d[0]][1]):int(a[d[0]][1]+sl), int(a[d[0]][0]):int(a[d[0]][0]+sl)]

            # crop_rectangle to crop_square (central_point:center point),as the
            # input of yolov5 must be square

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
            # 保存带有标注信息的检测结果
            #annotated_frame = annotate(image_source=image_source, boxes=original_boxes, logits=logits,
                                       #phrases=phrases)

            #crop_img = cv2.resize(crop_img, (640, 640)) #save GPU memory->SAM

            # 保存裁剪后的图像
            output_crop_path = os.path.join(output_folder_yellow_pan, f"{image_name.split('.')[0]}.jpg")
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_crop_path, crop_img)



        # Filter out only 'wood stick'
        filtered_boxes = []
        filtered_phrases = []
        for box, phrase in zip(boxes_filt, pred_phrases):
            if 'wood stick' in phrase:
                filtered_boxes.append(box)
                filtered_phrases.append(phrase)
        #print('filtered_phrases:',filtered_phrases)
        #print('filtered_boxes:',filtered_boxes)

        # 1. no 'wood stick'
        if len(filtered_boxes) == 0:
            print(f"No 'wood stick' found in {image_name}")
            #img = cv2.imread(input_image_path)
            #img = crop_img
            mask_stirring_tool=None

            """
            # 直至yolov5进行检测和输出对应的预测txt文件再执行下面的分析
            # label 路径
            label_path = os.path.join(labels_dir, f"{image_name.split('.')[0]}.txt")
            # 等待文件生成
            print(f"[INFO] Waiting for label file: {label_path}")
            wait_for_complete_file(label_path)
            time.sleep(1)
            # while not os.path.exists(label_path):
            # time.sleep(0.1)  # 每 10ms 检查一次
            # 一旦存在，执行分析
            print(f"[INFO] Found label file. Running AphidAnalyzer...")
            """

            # 直至yolov5进行检测和输出对应的预测txt文件再执行下面的分析 #
            label_path = os.path.join(labels_dir, f"{image_name.split('.')[0]}.txt")
            print(f"[INFO] Waiting for YOLOv5 to complete detection for {image_name}...")
            wait_for_complete_file(label_path)
            print(f"[INFO] Found label file. Running AphidAnalyzer...")


            ## 图像处理 ##
            filtered_centroids = analysis(crop_img, image_name, output_folder_color, output_folder_contour,
                                          output_folder_heatmap, output_folder_gradient,
                                          kernel_dilate, kernel_erode, grid_size, mask_stirring_tool)
            #print('filtered_centroids:', filtered_centroids)

            if len(filtered_centroids) > 1:
                ## 密度分析 ##
                # Load image to get dimensions
                #image = cv2.imread(input_image_path)
                #image = crop_img
                img_height, img_width = crop_img.shape[:2]

                # Convert filtered_centroids to numpy array
                centroids = np.array(filtered_centroids)


                # Initialize analyzer
                analyzer = AphidAnalyzer(centroids, (img_width, img_height),
                                         os.path.basename(input_image_path).split('.')[0],labels_dir)
                print(f"\n=== Analyzing image: {os.path.basename(input_image_path)} ===")
                print(f"Automatically set parameters: {analyzer.params}")

                # Run full analysis
                analyzer.analyze_distribution()

                # 记录结束时间
                end_time = time.time()
                # 计算运行时间
                elapsed_time = end_time - start_time
                print(f"Running time：{elapsed_time:.6f} S")
                time_records.append(elapsed_time)
                print("time_records:", time_records)

            continue

        # 2. yes 'wood stick' #相对原始图像
        """
        filtered_boxes = torch.stack(filtered_boxes)
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(filtered_boxes.size(0)):
            filtered_boxes[i] = filtered_boxes[i] * torch.Tensor([W, H, W, H])
            filtered_boxes[i][:2] -= filtered_boxes[i][2:] / 2
            filtered_boxes[i][2:] += filtered_boxes[i][:2]
        filtered_boxes = filtered_boxes.cpu()


        # Prepare for SAM prediction
        image_cv = cv2.imread(input_image_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_cv)

        transformed_boxes = predictor.transform.apply_boxes_torch(filtered_boxes, image_cv.shape[:2]).to(device)

        # Predict masks using SAM
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        #print('type(masks):', type(masks))

        # 如果需要将 largest_mask 转回 torch 格式：
        # largest_mask_tensor = torch.from_numpy(largest_mask)
        save_mask_data(args.output_dir, masks, filtered_boxes, filtered_phrases, image_name)

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

        # Save box and yellow_mask
        output_image_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_output.jpg")
        img,mask_stirring_tool = save_output_with_opencv(image_cv, largest_mask, filtered_boxes, filtered_phrases, output_image_path)

        # Save mask data
        output_image_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_largest_mask.jpg")
        cv2.imwrite(output_image_path, largest_mask * 255)
        """


        # 2. yes 'wood stick' #只提取yellow_pan区域的wood stick
        filtered_boxes = torch.stack(filtered_boxes)
        size = image_pil.size
        H, W = size[1], size[0]

        for i in range(filtered_boxes.size(0)):
            filtered_boxes[i] = filtered_boxes[i] * torch.Tensor([W, H, W, H])
            filtered_boxes[i][:2] -= filtered_boxes[i][2:] / 2
            filtered_boxes[i][2:] += filtered_boxes[i][:2]

        # 将 'wood stick' box 转换为 crop_img 中的坐标
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
        image_cv = crop_img  # ✅ 使用裁剪图作为输入图像
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # 直至yolov5进行检测和输出对应的预测txt文件再执行下面的聚类分析
        #label_path = os.path.join(labels_dir, f"{image_name.split('.')[0]}.txt")
        # 等待文件生成
        #print(f"[INFO] Waiting for label file: {label_path}")
        #while not os.path.exists(label_path):
            #time.sleep(0.01)  # 每 10ms 检查一次
        # 一旦存在，执行分析
        #print(f"[INFO] Found label file. Running AphidAnalyzer...")

        ## 直至yolov5进行检测和输出对应的预测txt文件再执行下面的聚类分析 ##
        # 1. 首先等待YOLOv5完成检测
        label_path = os.path.join(labels_dir, f"{image_name.split('.')[0]}.txt")
        print(f"[INFO] Waiting for YOLOv5 to complete detection for {image_name}...")
        wait_for_complete_file(label_path)
        print(f"[INFO] Found label file. Running AphidAnalyzer...")


        # 2. 确保释放YOLOv5占用的GPU资源
        #time.sleep(1)  # 给YOLOv5时间释放资源
        torch.cuda.empty_cache()

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


        # 如果需要将 largest_mask 转回 torch 格式：
        # largest_mask_tensor = torch.from_numpy(largest_mask)
        save_mask_data(args.output_dir, masks, filtered_boxes_crop, filtered_phrases, image_name)

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

        # Save box and yellow_mask
        output_image_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_output.jpg")
        img, mask_stirring_tool = save_output_with_opencv(image_cv, largest_mask, filtered_boxes_crop, filtered_phrases,
                                                          output_image_path)

        # Save mask data
        output_image_path = os.path.join(output_dir, f"{image_name.split('.')[0]}_largest_mask.jpg")
        cv2.imwrite(output_image_path, largest_mask * 255)







        ## 图像处理 ##
        filtered_centroids=analysis(crop_img, image_name, output_folder_color, output_folder_contour,
                 output_folder_heatmap, output_folder_gradient,
                 kernel_dilate, kernel_erode, grid_size,mask_stirring_tool)

        #print('filtered_centroids:',filtered_centroids)


        if len(filtered_centroids) > 1:
            ## 密度分析 ##
            # Load image to get dimensions
            #image = cv2.imread(input_image_path)
            #img_height, img_width = image.shape[:2]
            img_height, img_width = crop_img.shape[:2]

            # Convert filtered_centroids to numpy array
            centroids = np.array(filtered_centroids)

            # Initialize analyzer
            analyzer = AphidAnalyzer(centroids, (img_width, img_height), os.path.basename(input_image_path).split('.')[0],labels_dir)
            print(f"\n=== Analyzing image: {os.path.basename(input_image_path)} ===")
            print(f"Automatically set parameters: {analyzer.params}")

            # Run full analysis
            analyzer.analyze_distribution()

            # 记录结束时间
            end_time = time.time()
            # 计算运行时间
            elapsed_time = end_time - start_time
            print(f"Running time：{elapsed_time:.6f} S")
            time_records.append(elapsed_time)
            print("time_records:", time_records)

        # 确保释放模型内存
        #del model
        #del sam
        #del predictor
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
    args = parser.parse_args()
    # 定义 YOLO 标签文件目录
    #labels_dir = "/home/newdrive/Phd/Modification_yolov5/runs/detect/simulation_test_soft_nms/labels"
    labels_dir = "./runs/detect/counting_confidence/labels"

    process_directory(args.input_dir, args.output_dir, args)























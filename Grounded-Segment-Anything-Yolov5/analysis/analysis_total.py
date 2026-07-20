import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


# 定义形状类型和对应的颜色
SHAPES = ['round', 'square', 'triangle', 'spiral', 'four_circles', 'random_lines']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']


# 定义时间点（1-25对应2-50秒，间隔2秒）
time_full = list(range(1, 27))  # 完整序列：1-25
time_cut = list(range(10, 27))  # 截取序列：10-25 (对应18-50秒)

# 定义R²权重（根据实际情况调整）
r2_weights = {
    "PDU": 0.1135,
    "MDCBB": 0.1524,
    "AGM": 0.1737,
    "IC": 0.2559,
    "IQA": 0.2722,
    "PN": 0.2869

    #"TP": 0,
    #"FP": 0,
    #"FN": 0,
    #"Counting_confidence":0,
    #"GT_image": 0,
    #"GT_real": 0
}

# 归一化权重
total_weight = sum(r2_weights.values())
weights = {k: v / total_weight for k, v in r2_weights.items()}

# 定义哪些指标是越小越好
negative_metrics = ['IQA', 'IC', 'PDU', 'PN']


# 1. 读取并合并数据
def load_and_merge_data(records_file, counting_file):
    # 读取result_records文件
    records_df = pd.read_csv(records_file, header=None,
                             names=['Image Name', 'MDCBB', 'PN', 'AGM', 'IC', 'IQA', 'PDU'])

    # 读取counting_result文件
    counting_df = pd.read_csv(counting_file, header=None,
                              names=['Image Name', 'TP', 'FP', 'FN', 'Counting_confidence', 'GT_image', 'GT_real'])

    # 合并两个DataFrame
    merged_df = pd.merge(records_df, counting_df, on='Image Name', how='inner')

    return merged_df


# 2. 解析图像名称信息
def parse_image_name(name):
    # 示例名称: low_four_circles_11_frame_000002.jpg
    #match = re.match(r"low_(\w+)_(\d+)_frame_(\d+)\.jpg", name)
    #match = re.match(r"med_(\w+)_(\d+)_frame_(\d+)\.jpg", name)
    match = re.match(r"high_(\w+)_(\d+)_frame_(\d+)\.jpg", name)
    if match:
        shape_type = match.group(1)  # 形状类型，如four_circles
        subgroup = match.group(2)  # 子组编号，如11
        time_point = int(match.group(3))  # 时间点，如2 (对应4秒)
        return shape_type, subgroup, time_point
    return None, None, None


# 3. 主分析函数
def analyze_data(merged_df):
    # 解析图像名称信息
    merged_df[['Shape', 'Subgroup', 'Time']] = merged_df['Image Name'].apply(
        lambda x: pd.Series(parse_image_name(x)))

    # 转换Time为实际秒数 (每帧间隔2秒)
    merged_df['Time_seconds'] = merged_df['Time'] * 2

    # 定义所有指标及其显示名称
    metrics = {
        # 来自result_records的指标
        'MDCBB': 'MDCBB',
        'PN': 'PN',
        'AGM': 'AGM',
        'IC': 'IC',
        'IQA': 'IQA',
        'PDU': 'PDU',
        # 来自counting_result的指标
        'TP': 'TP',
        'FP': 'FP',
        'FN': 'FN',
        'Counting_confidence': 'Counting Confidence',
        'GT_image': 'GT_image',
        'GT_real': 'GT_real'
    }

    # 为每个指标创建单独的图表
    for metric, display_name in metrics.items():
        plt.figure(figsize=(12, 8))

        # 对每个形状类型绘制曲线
        for shape, color in zip(SHAPES, COLORS):
            # 筛选当前形状的数据
            shape_data = merged_df[merged_df['Shape'] == shape]

            # 计算每个时间点的均值
            avg_values = shape_data.groupby('Time_seconds')[metric].mean()

            # 绘制曲线
            sns.lineplot(x=avg_values.index, y=avg_values.values,
                         label=f'{shape}', color=color, marker='o', markersize=6)

            # 打印均值序列
            print(f"\n===== Mean values of {shape} type {display_name} over time =====")
            print(avg_values.round(4).to_string())

        # 设置图表属性
        plt.title(f'{display_name} over time (comparison across all shape types)')
        plt.xlabel('Time (s)')
        plt.ylabel(display_name)
        plt.legend(title='Shape Type')
        plt.grid(True)

        # 调整x轴刻度为每5秒一个标记
        plt.xticks(range(0, 51, 5))

        plt.tight_layout()
        plt.show()



# 1. 对完整序列的分析 ==============================================
def analyze_full_sequence(merged_df):
    print("\n" + "=" * 50)
    print("Full sequence analysis (0–50 seconds)")
    print("=" * 50)

    # 计算每个形状类型每个指标的均值
    shape_metrics = {}
    for shape in SHAPES:
        shape_data = merged_df[merged_df['Shape'] == shape]
        metrics_avg = shape_data.groupby('Shape')[list(r2_weights.keys())].mean().iloc[0]
        shape_metrics[shape] = metrics_avg

    # 计算加权得分
    weighted_scores = {shape: 0 for shape in SHAPES}
    for shape in SHAPES:
        for metric, weight in weights.items():
            value = shape_metrics[shape][metric]
            if metric in negative_metrics:
                weighted_scores[shape] += (1 / value) * weight  # 越小越好，取倒数
                #weighted_scores[shape] -= value * weight  # 越小越好
            else:
                weighted_scores[shape] += value * weight  # 越大越好

    # 计算误差 (GT_real - TP)
    errors = {}
    for shape in SHAPES:
        shape_data = merged_df[merged_df['Shape'] == shape]
        avg_tp = shape_data['TP'].mean()
        avg_gt = shape_data['GT_real'].mean()
        errors[shape] = avg_gt - avg_tp

    # 计算综合得分 (加权得分 - 误差)
    alpha = 1.0  # 误差惩罚系数
    final_scores = {}
    for shape in SHAPES:
        final_scores[shape] = weighted_scores[shape] - alpha * errors[shape]

    # 打印结果
    print("\nAverage metrics for each shape type(0-50s):")
    for shape in SHAPES:
        print(f"\n{shape}:")
        for metric in r2_weights.keys():
            print(f"  {metric}: {shape_metrics[shape][metric]:.4f}")

    print("\nWeighted scores:")
    for shape, score in weighted_scores.items():
        print(f"  {shape}: {score:.4f}")

    print("\nError (GT_real - TP):")
    for shape, error in errors.items():
        print(f"  {shape}: {error:.4f}")

    print("\nFinal score (Weighted score - Error):")
    for shape, score in final_scores.items():
        print(f"  {shape}: {score:.4f}")

    # 找出最佳形状
    best_shape = max(final_scores.items(), key=lambda x: x[1])[0]
    print(f"\n✅ Based on full-sequence analysis, the best stirring shape is: {best_shape}")


# 2. 对截取序列的分析 (18-50秒) ==================================

def analyze_cut_sequence(merged_df):
    print("\n" + "=" * 50)
    print("Truncated sequence analysis (18–50 seconds)")
    print("=" * 50)

    # 筛选18-50秒的数据 (Time >= 9)
    cut_df = merged_df[merged_df['Time'] >= 9]

    # 计算每个形状类型每个指标的均值
    shape_metrics = {}
    for shape in SHAPES:
        shape_data = cut_df[cut_df['Shape'] == shape]
        metrics_avg = shape_data.groupby('Shape')[list(r2_weights.keys())].mean().iloc[0]
        shape_metrics[shape] = metrics_avg

    # 计算加权得分
    weighted_scores = {shape: 0 for shape in SHAPES}
    for shape in SHAPES:
        for metric, weight in weights.items():
            value = shape_metrics[shape][metric]
            if metric in negative_metrics:
                weighted_scores[shape] += (1 / value) * weight #越小越好
                #weighted_scores[shape] -= value * weight #越小越好

            else:
                weighted_scores[shape] += value * weight

    # 计算误差 (GT_real - TP)
    errors = {}
    for shape in SHAPES:
        shape_data = cut_df[cut_df['Shape'] == shape]
        avg_tp = shape_data['TP'].mean()
        avg_gt = shape_data['GT_real'].mean()
        errors[shape] = avg_gt - avg_tp

    # 计算综合得分 (加权得分 - 误差)
    alpha = 1.0  # 误差惩罚系数
    final_scores = {}
    for shape in SHAPES:
        final_scores[shape] = weighted_scores[shape] - alpha * errors[shape]

    # 打印结果
    print("\n\nAverage metrics for each shape type (18-50s):")
    for shape in SHAPES:
        print(f"\n{shape}:")
        for metric in r2_weights.keys():
            print(f"  {metric}: {shape_metrics[shape][metric]:.4f}")

    print("\nWeighted scores (18-50s):")
    for shape, score in weighted_scores.items():
        print(f"  {shape}: {score:.4f}")

    print("\nError (GT_real - TP) (18-50s):")
    for shape, error in errors.items():
        print(f"  {shape}: {error:.4f}")

    print("\nFinal score (Weighted score - Error) (18-50s):")
    for shape, score in final_scores.items():
        print(f"  {shape}: {score:.4f}")

    # 找出最佳形状
    best_shape = max(final_scores.items(), key=lambda x: x[1])[0]
    print(f"\n✅ Based on full-sequence analysis, the best stirring shape is: {best_shape}")

    # 绘制截取序列的指标变化曲线
    for metric in r2_weights.keys():
        plt.figure(figsize=(12, 6))

        for shape, color in zip(SHAPES, COLORS):
            # 筛选当前形状和截取时间的数据
            shape_data = merged_df[(merged_df['Shape'] == shape) &
                                   (merged_df['Time'] >= 9)]

            # 计算每个时间点的均值
            avg_values = shape_data.groupby('Time_seconds')[metric].mean()

            # 绘制曲线
            sns.lineplot(x=avg_values.index, y=avg_values.values,
                         label=shape, color=color, marker='o', markersize=6)

        plt.title(f'{metric} over time (18-50s, all shape types)')
        plt.xlabel('Time (s)')
        plt.ylabel(metric)
        plt.legend(title='Shape Type')
        plt.grid(True)
        plt.xticks(range(18, 51, 2))
        plt.tight_layout()
        plt.show()


def analyze_full_sequence_counting_confidence(merged_df):
    print("\n" + "=" * 50)
    print("Full sequence analysis using Counting Confidence and Error (0-50 seconds)")
    print("=" * 50)

    # 计算每个形状类型的Counting Confidence均值序列的均值
    counting_confidences = {}
    for shape in SHAPES:
        shape_data = merged_df[merged_df['Shape'] == shape]
        # 先计算随时间变化的均值序列
        avg_series = shape_data.groupby('Time_seconds')['Counting_confidence'].mean()
        # 再计算整个序列的均值
        counting_confidences[shape] = avg_series.mean()

    # 计算误差 (GT_real - TP)
    errors = {}
    for shape in SHAPES:
        shape_data = merged_df[merged_df['Shape'] == shape]
        avg_tp = shape_data['TP'].mean()
        avg_gt = shape_data['GT_real'].mean()
        errors[shape] = avg_gt - avg_tp

    # 计算综合得分 (Counting Confidence均值 - 误差)
    alpha = 1.0  # 误差惩罚系数
    final_scores = {}
    for shape in SHAPES:
        final_scores[shape] = counting_confidences[shape] - alpha * errors[shape]

    # 打印结果
    print("\nAverage Counting Confidence for each shape type (0-50s):")
    for shape, cc in counting_confidences.items():
        print(f"  {shape}: {cc:.4f}")

    print("\nError (GT_real - TP):")
    for shape, error in errors.items():
        print(f"  {shape}: {error:.4f}")

    print("\nFinal score (Counting Confidence - Error):")
    for shape, score in final_scores.items():
        print(f"  {shape}: {score:.4f}")

    # 找出最佳形状
    best_shape = max(final_scores.items(), key=lambda x: x[1])[0]
    print(f"\n✅ Based on Counting Confidence and Error analysis (0-50s), the best stirring shape is: {best_shape}")


# 4. 使用Counting Confidence替代加权得分的截取序列分析 =====================

def analyze_cut_sequence_counting_confidence(merged_df):
    print("\n" + "=" * 50)
    print("Truncated sequence analysis using Counting Confidence and Error (18-50 seconds)")
    print("=" * 50)

    # 筛选18-50秒的数据 (Time >= 9)
    cut_df = merged_df[merged_df['Time'] >= 9]

    # 计算每个形状类型的Counting Confidence均值序列的均值
    counting_confidences = {}
    for shape in SHAPES:
        shape_data = cut_df[cut_df['Shape'] == shape]
        # 先计算随时间变化的均值序列
        avg_series = shape_data.groupby('Time_seconds')['Counting_confidence'].mean()
        # 再计算整个序列的均值
        counting_confidences[shape] = avg_series.mean()

    # 计算误差 (GT_real - TP)
    errors = {}
    for shape in SHAPES:
        shape_data = cut_df[cut_df['Shape'] == shape]
        avg_tp = shape_data['TP'].mean()
        avg_gt = shape_data['GT_real'].mean()
        errors[shape] = avg_gt - avg_tp

    # 计算综合得分 (Counting Confidence均值 - 误差)
    alpha = 1.0  # 误差惩罚系数
    final_scores = {}
    for shape in SHAPES:
        final_scores[shape] = counting_confidences[shape] - alpha * errors[shape]

    # 打印结果
    print("\nAverage Counting Confidence for each shape type (18-50s):")
    for shape, cc in counting_confidences.items():
        print(f"  {shape}: {cc:.4f}")

    print("\nError (GT_real - TP) (18-50s):")
    for shape, error in errors.items():
        print(f"  {shape}: {error:.4f}")

    print("\nFinal score (Counting Confidence - Error) (18-50s):")
    for shape, score in final_scores.items():
        print(f"  {shape}: {score:.4f}")

    # 找出最佳形状
    best_shape = max(final_scores.items(), key=lambda x: x[1])[0]
    print(f"\n✅ Based on Counting Confidence and Error analysis (18-50s), the best stirring shape is: {best_shape}")

    # 绘制截取序列的Counting Confidence变化曲线
    plt.figure(figsize=(12, 6))
    for shape, color in zip(SHAPES, COLORS):
        # 筛选当前形状和截取时间的数据
        shape_data = merged_df[(merged_df['Shape'] == shape) &
                               (merged_df['Time'] >= 9)]

        # 计算每个时间点的均值
        avg_values = shape_data.groupby('Time_seconds')['Counting_confidence'].mean()

        # 绘制曲线
        sns.lineplot(x=avg_values.index, y=avg_values.values,
                     label=shape, color=color, marker='o', markersize=6)

    plt.title('Counting Confidence over time (18-50s, all shape types)')
    plt.xlabel('Time (s)')
    plt.ylabel('Counting Confidence')
    plt.legend(title='Shape Type')
    plt.grid(True)
    plt.xticks(range(18, 51, 2))
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 文件路径 - 请根据实际情况修改
    records_file = "result_records_high_density_shapes.txt"
    counting_file = "counting_result_information_high_density_shapes_sorted.txt"

    # 检查文件是否存在
    if not os.path.exists(records_file):
        print(f"Error: file {records_file} not exist!")
        exit(1)
    if not os.path.exists(counting_file):
        print(f"Error: file {counting_file} not exist!")
        exit(1)

    # 加载和合并数据
    print("loading and merging data...")
    merged_df = load_and_merge_data(records_file, counting_file)

    # 分析数据并绘制图表
    print("analying and plotting...")
    analyze_data(merged_df)



    # 确保merged_df已从主程序加载
    if 'merged_df' in globals():

        ##基于6个指标的加权得分 + Error来做出判断##
        # 分析完整序列
        analyze_full_sequence(merged_df)
        # 分析截取序列
        analyze_cut_sequence(merged_df)



        ##基于Counting_confidence + Error来做出判断##
        analyze_full_sequence_counting_confidence(merged_df)
        analyze_cut_sequence_counting_confidence(merged_df)


    else:
        print("Error: not find merged_df")

















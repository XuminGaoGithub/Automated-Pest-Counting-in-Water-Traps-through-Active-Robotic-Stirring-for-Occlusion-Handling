from collections import defaultdict
import re

# 输入和输出文件路径
input_path = 'counting_result_information_high_density_speed.txt'
output_path = 'counting_result_information_high_density_speed_sorted.txt'

# 提取组名函数，例如从 low_four_circles_1_frame_000000.jpg 提取 low_four_circles_1
def extract_group_key(filename):
    match = re.match(r"(.+?)_frame_\d{6}\.jpg", filename)
    return match.group(1) if match else None

# 提取帧号函数，例如从 frame_000003 提取 3
def extract_frame_number(filename):
    match = re.search(r"_frame_(\d{6})\.jpg", filename)
    return int(match.group(1)) if match else -1

# 分组存储
groups = defaultdict(list)

# 读取原始数据
with open(input_path, 'r') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        filename = line.split(',')[0]
        group_key = extract_group_key(filename)
        frame_number = extract_frame_number(filename)
        if group_key is not None and frame_number != -1:
            groups[group_key].append((frame_number, line))

# 排序后写入输出文件
with open(output_path, 'w') as out:
    for group_name in sorted(groups.keys()):
        sorted_lines = sorted(groups[group_name], key=lambda x: x[0])
        for _, line in sorted_lines:
            out.write(line + '\n')

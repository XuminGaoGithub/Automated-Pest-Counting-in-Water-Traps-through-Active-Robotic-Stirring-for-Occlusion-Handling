# refer to https://github.com/z1069614715/objectdetection_script/blob/master/objectdetection-tricks/tricks_1.py
"""
import os, cv2, tqdm, shutil
import numpy as np

def xywh2xyxy(box):
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box

def iou(box1, box2):

    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
 
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
 
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
 
    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter
 
    iou = area_inter / area_union
    return iou

def draw_box(img, box, color):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=5)
    return img

if __name__ == '__main__':
    postfix = 'jpg'
    predictfix = 'txt'
    labelfix = 'txt'
    img_path = '/home/newdrive/simulation/test/test_20241120/yellow_pan/Aphid_40_soil_00_highspeed/'
    label_path = '/home/newdrive/Phd/Modification_yolov5/dataset/voc2065/labels/Aphid_40_soil_00_highspeed/'
    predict_path = '/home/newdrive/Phd/Modification_yolov5/runs/detect/Aphid_40_soil_00_highspeed/labels/'
    #predict_path = '/home/newdrive/Phd/Modification_yolov5/runs/detect/exp/labels/'
    save_path = '/home/newdrive/Phd/Modification_yolov5/runs/vis_tpfpfn/'

    classes = ['aphid']
    detect_color, missing_color, error_color  = (0, 255, 0), (0, 0, 255), (255, 0, 0)
    iou_threshold = 0.5
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    all_right_num, all_missing_num, all_error_num = 0, 0, 0
    with open('result.txt', 'w') as f_w:
        for path in tqdm.tqdm(os.listdir(label_path)):
            print('path:',path)
            print(f'{img_path}/{path[:-4]}.{postfix}')
            image = cv2.imread(f'{img_path}/{path[:-4]}.{postfix}')
            if image is None:
                print(f'image:{img_path}/{path[:-4]}.{postfix} not found.', file=f_w)
            h, w = image.shape[:2]
            
            try:
                with open(f'{predict_path}/{path[:-4]}.{predictfix}') as f:
                    print(f'{predict_path}/{path[:-4]}.{predictfix}')
                    #print('yes')
                    pred = np.array(list(map(lambda x:np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                    print('pred:', pred)
                    pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])
                    pred[:, [1, 3]] *= w
                    pred[:, [2, 4]] *= h
                    pred = list(pred)
            except:
                pred = []
                print('no_pred')
            
            try:
                with open(f'{label_path}/{path[:-4]}.{labelfix}') as f:
                    print(f'{label_path}/{path[:-4]}.{labelfix}')
                    label = np.array(list(map(lambda x:np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                    print('label:', label)
                    label[:, 1:] = xywh2xyxy(label[:, 1:])
                    label[:, [1, 3]] *= w
                    label[:, [2, 4]] *= h
            except:
                print(f'label path:{label_path}/{path} (not found or no target).', file=f_w)
                label = None

            if label is not None:
                right_num, missing_num, error_num = 0, 0, 0
                label_id, pred_id = list(range(label.shape[0])), [] if len(pred) == 0 else list(range(len(pred)))
                for i in range(label.shape[0]):
                    if len(pred) == 0: break
                    ious = iou(label[i:i+1, 1:], np.array(pred)[:, 1:5])[0]
                    print('ious:',ious)
                    ious_argsort = ious.argsort()[::-1]
                    missing = True
                    for j in ious_argsort:
                        if ious[j] < iou_threshold:
                            print('ious[j] < iou_threshold')
                            break
                        if label[i, 0] == pred[j][0]:
                            image = draw_box(image, pred[j][1:5], detect_color)
                            pred.pop(j)
                            missing = False
                            right_num += 1
                            break

                    if missing:
                        image = draw_box(image, label[i][1:5], missing_color)
                        missing_num += 1

                if len(pred):
                    for j in range(len(pred)):
                        image = draw_box(image, pred[j][1:5], error_color)
                        error_num += 1

                all_right_num, all_missing_num, all_error_num = all_right_num + right_num, all_missing_num + missing_num, all_error_num + error_num
                image = cv2.resize(image, (2000, 2000))
                cv2.imwrite(f'{save_path}/{path[:-4]}.{postfix}', image)
                print(f'name:{path[:-4]} right:{right_num} missing:{missing_num} error:{error_num}', file=f_w)
        print(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}', file=f_w)
"""


"""
#上面的代码有问题，不能正确绘制Fn(对于soft-nms的预测结果)，下面的代码修复了这一问题
import os
import cv2
import tqdm
import shutil
import numpy as np

def xywh2xyxy(box):
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box

def iou(box1, box2):
    if len(box2) == 0:
        return np.zeros((box1.shape[0], 1))  # 如果没有预测框，返回全0
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter
    iou = area_inter / area_union
    return iou

def draw_box(img, box, color):
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=5)
    return img

if __name__ == '__main__':
    postfix = 'jpg'
    predictfix = 'txt'
    labelfix = 'txt'
    #img_path = '/home/newdrive/Phd/Modification_yolov5/dataset/voc2066/JPEGImages/'
    #label_path = '/home/newdrive/Phd/Modification_yolov5/dataset/voc2066/labels/'
    #predict_path = '/home/newdrive/Phd/Modification_yolov5/runs/detect/simulation_test_soft_nms/labels/'
    #save_path = '/home/newdrive/simulation/test/confidence_analysis/vis_TPFPFN/soft_nms/'

    img_path = '/home/newdrive/Phd/Modification_yolov5/dataset/voc2067/JPEGImages/'
    label_path = '/home/newdrive/Phd/Modification_yolov5/dataset/voc2067/labels/'
    # predict_path = '/home/newdrive/Phd/Modification_yolov5/runs/detect/Aphid_40_soil_00_highspeed/labels/'
    predict_path = '/home/newdrive/Phd/Modification_yolov5/runs/detect/simulation_test_soft-nms-aphid_40_otherinsects_v2/labels/'
    save_path = '/home/newdrive/simulation/test/confidence_analysis/vis_TPFPFN/soft_nms_aphid_otherinsects_v2/'

    classes = ['aphid']
    detect_color, missing_color, error_color = (0, 255, 0), (0, 0, 255), (255, 0, 0)
    iou_threshold = 0.5

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    all_right_num, all_missing_num, all_error_num = 0, 0, 0
    with open('result.txt', 'w') as f_w:
        for path in tqdm.tqdm(os.listdir(label_path)):
            print('path:', path)
            image = cv2.imread(f'{img_path}/{path[:-4]}.{postfix}')
            if image is None:
                print(f'image:{img_path}/{path[:-4]}.{postfix} not found.', file=f_w)
                continue
            h, w = image.shape[:2]

            # 读取预测框
            try:
                with open(f'{predict_path}/{path[:-4]}.{predictfix}') as f:
                    pred = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                    pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])
                    pred[:, [1, 3]] *= w
                    pred[:, [2, 4]] *= h
                    pred = list(pred)
            except:
                pred = []
                print('no_pred')

            # 读取真实框
            try:
                with open(f'{label_path}/{path[:-4]}.{labelfix}') as f:
                    label = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                    label[:, 1:] = xywh2xyxy(label[:, 1:])
                    label[:, [1, 3]] *= w
                    label[:, [2, 4]] *= h
            except:
                print(f'label path:{label_path}/{path} (not found or no target).', file=f_w)
                label = None

            if label is not None:
                right_num, missing_num, error_num = 0, 0, 0

                # 遍历每个真实框
                for i in range(label.shape[0]):
                    if len(pred) == 0:
                        # 如果没有预测框，所有真实框都是 FN
                        image = draw_box(image, label[i][1:5], missing_color)
                        missing_num += 1
                        continue

                    # 计算当前真实框与所有预测框的 IoU
                    ious = iou(label[i:i+1, 1:], np.array(pred)[:, 1:5])[0]
                    print(f"Real box {i} IoUs: {ious}")  # 打印每个真实框的 IoU 值

                    if len(ious) == 0:
                        # 如果没有匹配的预测框，标记为 FN
                        image = draw_box(image, label[i][1:5], missing_color)
                        missing_num += 1
                        continue

                    # 找到 IoU 最大的预测框
                    max_iou_idx = np.argmax(ious)
                    if ious[max_iou_idx] >= iou_threshold and label[i, 0] == pred[max_iou_idx][0]:
                        # 如果 IoU 大于阈值且类别匹配，标记为 TP
                        image = draw_box(image, pred[max_iou_idx][1:5], detect_color)
                        pred.pop(max_iou_idx)  # 移除已匹配的预测框
                        right_num += 1
                    else:
                        # 否则标记为 FN
                        image = draw_box(image, label[i][1:5], missing_color)
                        missing_num += 1

                # 处理剩余的预测框（FP）
                for j in range(len(pred)):
                    image = draw_box(image, pred[j][1:5], error_color)
                    error_num += 1

                # 更新统计结果
                all_right_num += right_num
                all_missing_num += missing_num
                all_error_num += error_num

                # 保存结果图像
                image = cv2.resize(image, (2000, 2000))
                cv2.imwrite(f'{save_path}/{path[:-4]}.{postfix}', image)
                print(f'name:{path[:-4]} right:{right_num} missing:{missing_num} error:{error_num}', file=f_w)

        # 输出最终统计结果
        print(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}', file=f_w)
"""



import os
import cv2
import tqdm
import shutil
import numpy as np
import xml.etree.ElementTree as ET
import torch

def xywh2xyxy(box):
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box

def iou(box1, box2):
    if len(box2) == 0:
        return np.zeros((box1.shape[0], 1))  # 如果没有预测框，返回全0
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)
    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))
    area_inter = np.maximum(0, (xb - xa + 1)) * np.maximum(0, (yb - ya + 1))
    area_1 = (x12 - x11 + 1) * (y12 - y11 + 1)
    area_2 = (x22 - x21 + 1) * (y22 - y21 + 1)
    area_union = area_1 + np.transpose(area_2) - area_inter
    iou = area_inter / area_union
    return iou

def draw_box(img, box, color, confidence=None):
    """
    绘制检测框，并显示置信度（如果提供）。
    :param img: 输入图像
    :param box: 检测框坐标 [x1, y1, x2, y2]
    :param color: 检测框颜色
    :param confidence: 置信度值（可选）
    """
    cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, thickness=3)
    if confidence is not None:
        # 在检测框左上角显示置信度
        text = f"{confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = int(box[0])
        text_y = int(box[1]) - 10 if int(box[1]) - 10 > 10 else int(box[1]) + 20
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    return img

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = []

    for obj in root.findall('object'):
        label = []

        # Assuming there is only one class named 'aphid'
        label.append(0)  # Class index for 'aphid'

        bbox = obj.find('bndbox')
        label.append(float(bbox.find('xmin').text))
        label.append(float(bbox.find('ymin').text))
        label.append(float(bbox.find('xmax').text))
        label.append(float(bbox.find('ymax').text))

        labels.append(label)

    return torch.tensor(labels).to('cuda'), len(labels)



if __name__ == '__main__':
    postfix = 'jpg'
    predictfix = 'txt'
    labelfix = 'txt'
    xmlfix = 'xml'

    img_path = './dataset/voc2077/JPEGImages/'
    Annotations_path = './dataset/voc2077/Annotations' #gt_Annotations_VOC_format
    label_path = './dataset/voc2077/labels/' #gt_Annotations_yolo_format
    predict_path = '/home/newdrive/Phd/Grounded-Segment-Anything-Yolov5/Demo_video/shape_speed/counting_confidence/pre_labels' #"'./runs/detect/counting_confidence_high_density_speed_without_conf/pred_labels' #pred_labels
    save_path = '/home/newdrive/Phd/Grounded-Segment-Anything-Yolov5/Demo_video/shape_speed/vis_TPFPFN'


    #img_path = '/home/newdrive/Phd/Modification_yolov5/dataset/voc2067/JPEGImages/'
    #label_path = '/home/newdrive/Phd/Modification_yolov5/dataset/voc2067/labels/'
    #predict_path = '/home/newdrive/Phd/Modification_yolov5/runs/detect/simulation_test_soft-nms-aphid_40_otherinsects_v2/labels/'
    #save_path = '/home/newdrive/simulation/test/confidence_analysis/vis_TPFPFN/soft_nms_aphid_otherinsects_v2_withscore/'

    classes = ['aphid']
    detect_color, missing_color, error_color = (0, 255, 0), (0, 0, 255), (255, 0, 0)
    iou_threshold = 0.6 #yolov5中使用的iou_thres=0.45

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    all_right_num, all_missing_num, all_error_num = 0, 0, 0
    with open('counting_result_information.txt', 'w') as f_w:
        for path in tqdm.tqdm(os.listdir(label_path)):
            print('\n')
            print('path:', path)
            image = cv2.imread(f'{img_path}/{path[:-4]}.{postfix}')
            if image is None:
                print(f'image:{img_path}/{path[:-4]}.{postfix} not found.', file=f_w)
                continue
            h, w = image.shape[:2]

            # 读取预测框
            try:
                with open(f'{predict_path}/{path[:-4]}.{predictfix}') as f:
                    pred = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                    pred[:, 1:5] = xywh2xyxy(pred[:, 1:5])
                    pred[:, [1, 3]] *= w
                    pred[:, [2, 4]] *= h
                    pred = list(pred)
                    total_pred_num = len(pred)
            except:
                pred = []
                print('no_pred')



            # 读取真实框
            try:
                with open(f'{label_path}/{path[:-4]}.{labelfix}') as f:
                    label = np.array(list(map(lambda x: np.array(x.strip().split(), dtype=np.float32), f.readlines())))
                    label[:, 1:] = xywh2xyxy(label[:, 1:])
                    label[:, [1, 3]] *= w
                    label[:, [2, 4]] *= h
            except:
                print(f'label path:{label_path}/{path} (not found or no target).', file=f_w)
                label = None

            if label is not None:
                right_num, missing_num, error_num = 0, 0, 0

                # 遍历每个真实框
                for i in range(label.shape[0]):
                    if len(pred) == 0:
                        # 如果没有预测框，所有真实框都是 FN
                        image = draw_box(image, label[i][1:5], missing_color)
                        missing_num += 1
                        continue

                    # 计算当前真实框与所有预测框的 IoU
                    ious = iou(label[i:i+1, 1:], np.array(pred)[:, 1:5])[0]
                    print(f"Real box {i} IoUs: {ious}")  # 打印每个真实框的 IoU 值

                    if len(ious) == 0:
                        # 如果没有匹配的预测框，标记为 FN
                        image = draw_box(image, label[i][1:5], missing_color)
                        missing_num += 1
                        continue

                    # 找到 IoU 最大的预测框
                    max_iou_idx = np.argmax(ious)
                    if ious[max_iou_idx] >= iou_threshold and label[i, 0] == pred[max_iou_idx][0]:
                        # 如果 IoU 大于阈值且类别匹配，标记为 TP
                        #confidence = pred[max_iou_idx][5]  # 获取置信度
                        confidence = None
                        image = draw_box(image, pred[max_iou_idx][1:5], detect_color, confidence)
                        pred.pop(max_iou_idx)  # 移除已匹配的预测框
                        right_num += 1
                    else:
                        # 否则标记为 FN
                        image = draw_box(image, label[i][1:5], missing_color)
                        missing_num += 1

                # 处理剩余的预测框（FP）
                for j in range(len(pred)):
                    #confidence = pred[j][5]  # 获取置信度
                    confidence = None
                    image = draw_box(image, pred[j][1:5], error_color, confidence)
                    error_num += 1

                # 更新统计结果
                all_right_num += right_num
                all_missing_num += missing_num
                all_error_num += error_num

                # 保存结果图像
                image = cv2.resize(image, (2000, 2000))
                count_number = "Pred_Pests:{}".format(total_pred_num)
                cv2.putText(image, count_number, (160, 260), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

                xml_path = f'{Annotations_path}/{path[:-4]}.{xmlfix}'
                print('xml_path:',xml_path)
                label, gt_image = parse_xml(xml_path)
                print('gt_image:',gt_image)

                tp=right_num
                fp=error_num
                fn=missing_num
                Realiability = tp / (tp + fp + fn)
                gt_real = 12 #6，12，18 # gt number of aphids in the yellow pan
                print('gt_real:', gt_real)
                print(f"Realiability: {Realiability}")

                #print(f'name:{path[:-4]} right:{right_num} missing:{missing_num} error:{error_num}', file=f_w)
                print(f'{path[:-4]}.jpg,{tp},{fp},{fn},{Realiability:.4f},{gt_image},{gt_real}',
                      file=f_w)


                # 保存结果图像with counting confidence / Realiability
                count_confidence = f"Counting Error: {gt_real-tp}"
                cv2.putText(image, count_confidence, (160, 360), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
                count_confidence = f"Counting Confidence: {Realiability:.4f}"
                cv2.putText(image, count_confidence, (160, 460), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
                cv2.imwrite(f'{save_path}/{path[:-4]}.{postfix}', image)

        # 输出最终统计结果
        #print(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}', file=f_w)
        print(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}')
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
#refer to https://github.com/Hongyu-Yue/yoloV5_modify_smalltarget

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, soft_nms, print_args, scale_coords, strip_optimizer, xyxy2xywh,
                           apply_classifier_aphid)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from copy import deepcopy
import cv2
import copy
import torch.nn.functional as F



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

    return torch.tensor(labels).to('cuda')


def get_slice_bboxes(
    image_height=1000,
    image_width=1000,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
):
    """Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio(float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0
    #print(type(overlap_height_ratio))
    #print(type(slice_height))
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                #print('xmin, ymin, xmax, ymax:',xmin, ymin, xmax, ymax)
                #print('xmax-xmin,ymax-ymin:', xmax-xmin, ymax-ymin)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                #print('x_max-x_min,y_max-y_min:', x_max - x_min, y_max - y_min)
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def compute_and_visualize_gradients(image_path, output_image_path, output_blurriness_path):
    # 读取图像
    image = cv2.imread(image_path)
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

    # 可视化梯度图像
    grad_magnitude_display = cv2.convertScaleAbs(magnitude)

    # 保存梯度图像
    cv2.imwrite(output_image_path, grad_magnitude_display)

    # 绘制模糊度图像并保存
    plt.figure(figsize=(6, 6))
    plt.imshow(grad_magnitude_display, cmap='gray')
    plt.title('Gradient Magnitude')
    plt.colorbar()
    plt.savefig(output_blurriness_path)
    plt.close()

    return blurriness



@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        sliced_height=640, #256
        sliced_width=640, #256
        overlapped_height_ratio=0.2,
        overlapped_width_ratio=0.2,
        xml=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
):
    #line_thickness = 15
    nc = 1
    #confusion_matrix = ConfusionMatrix(nc=nc)
    source = str(source)
    img_pah = source
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    names = dict(enumerate(model.names if hasattr(model, 'names') else model.module.names))
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    #image = cv2.imread(img_pah)
    #imgsz = image.shape[0]
    #print('imgsz:',imgsz)

    classify = False
    if classify:
        modelc = torch.load('./dataset/voc2057_cls/model/resnet18_final.pth')  # initialize
        model.cuda()
        model.eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

    cnbr = open("cnbr.txt",'a')
    num=0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()

        # keep the original size rather than 640x640
        print('path:',path)
        file_name_with_extension = os.path.basename(path)
        #print('image:', file_name_with_extension)
        file_name = os.path.splitext(file_name_with_extension)[0]
        print('image:', file_name)

        #original_src = cv2.imread(img_pah)
        original_src = cv2.imread(path)
        original_src_copy = copy.deepcopy(original_src)

        #original_src = cv2.resize(original_src, (5123, 5123))

        ## for james_aphids
        #original_src = cv2.resize(original_src, (400, 400))
        #cv2.imwrite('1' + '.jpg', original_src)
        ##

        # 计算梯度模
        folder_path = '/home/newdrive/Phd/Modification_yolov5/runs/blurriness/'  # 替换为你的文件夹路径
        output_image_path = os.path.join(folder_path, f'{file_name}_grad.jpg')
        output_blurriness_path = os.path.join(folder_path, f'{file_name}_magnitude_plot.png')

        try:
            blurriness = compute_and_visualize_gradients(path, output_image_path, output_blurriness_path)
            print(f'Processed {file_name} - Blurriness: {blurriness}')
        except Exception as e:
            print(f'Error processing {file_name}: {e}')



        im = original_src
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        #print('im.shape', im.shape)


        im = torch.from_numpy(im).to(device)
        #print('im.size:',im.size())
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        nb, _, height, width = im.shape  # batch size, channels, height, width
        #print('height, width',height, width)

        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

        # split_merge straytage
        img = im
        mulpicplus = "2"  # 1 for normal_val,2 for SAHI_split_val
        assert (int(mulpicplus) >= 1)
        if mulpicplus == "1":
            pred, train_out = model(img, augment=augment, val=True)  # have to add train_out,otherwise error

        else:

            #图像对等份
            '''
            xsz = img.shape[2]
            ysz = img.shape[3]
            mulpicplus = int(mulpicplus)
            x_smalloccur = int(xsz / mulpicplus * 1.2)
            y_smalloccur = int(ysz / mulpicplus * 1.2)
            #print('x_smalloccur,y_smalloccur', x_smalloccur, y_smalloccur)
            for i in range(mulpicplus):
                x_startpoint = int(i * (xsz / mulpicplus))
                for j in range(mulpicplus):
                    y_startpoint = int(j * (ysz / mulpicplus))
                    x_real = min(x_startpoint + x_smalloccur, xsz)
                    y_real = min(y_startpoint + y_smalloccur, ysz)
                    if (x_real - x_startpoint) % 64 != 0:
                        x_real = x_real - (x_real - x_startpoint) % 64
                    if (y_real - y_startpoint) % 64 != 0:
                        y_real = y_real - (y_real - y_startpoint) % 64
                    dicsrc = img[:, :, x_startpoint:x_real,
                             y_startpoint:y_real]
                    pred_temp,train_out = model(dicsrc,
                                      augment=augment,
                                      val=True)
                    print('type(pred_temp):',type(pred_temp))
                    print('pred_temp:', pred_temp)
                    pred_temp[..., 0] = pred_temp[..., 0] + y_startpoint
                    pred_temp[..., 1] = pred_temp[..., 1] + x_startpoint
                    if i == 0 and j == 0:
                        out = pred_temp
                    else:
                        out = torch.cat([out, pred_temp], dim=1)
            '''

            slice_bboxes = get_slice_bboxes(
                image_height=height,
                image_width=width,
                slice_height=sliced_height,
                slice_width=sliced_width,
                overlap_height_ratio=overlapped_height_ratio,
                overlap_width_ratio=overlapped_width_ratio,
            )
            #print('slice_bboxes:', slice_bboxes)


            for i in range(len(slice_bboxes)):

                #for save the sliding window on the image
                #color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                         #(10, 100, 100), (100, 10, 100), (10, 100, 100)]

                #cv2.rectangle(img_copy, (slice_bboxes[i][0], slice_bboxes[i][1]),
                              #(slice_bboxes[i][2], slice_bboxes[i][3]), color[i], thickness=1, lineType=cv2.LINE_AA)


                #cropped_image = original_src[slice_bboxes[i][0]:slice_bboxes[i][2], slice_bboxes[i][1]:slice_bboxes[i][3]]
                cropped_image = copy.deepcopy(original_src[slice_bboxes[i][0]:slice_bboxes[i][2], slice_bboxes[i][1]:slice_bboxes[i][3]])
                cv2.imwrite('./runs/slicing/' + file_name + '_' + str(i) + '.jpg', cropped_image)
                #cropped_image_copy = cropped_image.copy()

                #for i in range(len(slice_bboxes)):
                x_startpoint = slice_bboxes[i][0]
                y_startpoint = slice_bboxes[i][1]
                x_endpoint = slice_bboxes[i][2]
                y_endpoint = slice_bboxes[i][3]

                dicsrc = img[:, :, x_startpoint:x_endpoint,
                         y_startpoint:y_endpoint]





                pred_temp, train_out = model(dicsrc,
                                             augment=augment,
                                             val=True)
                #print('type(pred_temp):', type(pred_temp))
                #print('pred_temp:', pred_temp)
                #print('pred_temp.szie:', pred_temp.size())

                #slice_pic=pred_temp
                slice_pic = pred_temp.clone()
                #print('pred_slice:', slice_pic)
                #print('slice_pic:',slice_pic)


                pred_temp[..., 0] = pred_temp[..., 0] + y_startpoint
                pred_temp[..., 1] = pred_temp[..., 1] + x_startpoint

                #print('iou_thres:', iou_thres)
                pred_slice = soft_nms(slice_pic, conf_thres, iou_thres, multi_label=True)  # Soft DIoU-NMS
                #print('type(pred_temp):', type(pred_slice))
                #print('pred_slice:', pred_slice)

                #print('type(cropped_image):', type(cropped_image))
                #print('cropped_image.shape:',cropped_image.shape)


                #save pediction results of the sliced block pics
                for j, det in enumerate(pred_slice):

                    annotator = Annotator(cropped_image, line_width=line_thickness, example=str(names))

                    if len(det):
                        #print('det[:, :4]:',det[:, :4])
                        #print('det[:, 0],det[:, 2],det[:, 1],det[:, 3]:',det[:, 0],det[:, 2],det[:, 1],det[:, 3])
                        # Subtract x_startpoint and y_startpoint
                        #det[:, 0] -= x_startpoint  # Subtract x_startpoint from the first x-coordinate
                        #det[:, 2] -= x_startpoint  # Subtract x_startpoint from the second x-coordinate
                        #det[:, 1] -= y_startpoint  # Subtract y_startpoint from the first y-coordinate
                        #det[:, 3] -= y_startpoint  # Subtract y_startpoint from the second y-coordinate
                        #print(f'det[:, :4]: {det[:, :4]}')

                        #det[:, :4] = scale_coords(im.shape[2:], det[:, :4], cropped_image.shape).round()
                        #print('im.shape[2:]', im.shape[2:])
                        #print('cropped_image.shape', cropped_image.shape)
                        #print('det[:, :4]:', det[:, :4])

                        for *xyxy, conf, cls in reversed(det):
                            #print('*xyxy, conf, cls:',*xyxy, conf, cls)

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                #print('color:', colors(c, True))
                                #cv2.imwrite('./runs/slicing/' + file_name + '_' + str(i) + '_pre.jpg', cropped_image)

                    cropped_image = annotator.result()
                    cv2.imwrite('./runs/slicing/' + file_name +'_'+str(i) + '_pred' + '.jpg', cropped_image)



                '''
                p1, p2 = (int(pred_temp[..., 0][i][0]), int(pred_temp[..., 0][i][1]), int(pred_temp[..., 1][i][2]), int(pred_temp[..., 1][i][3]))
                cv2.rectangle(cropped_image, p1, p2, (128, 128, 128), thickness=line_thickness, lineType=cv2.LINE_AA)
                cv2.imwrite('/home/xumin/yolov5/runs/slicing/' + str(i) + '.jpg', cropped_image)
                '''


                if i == 0:
                    out = pred_temp
                else:
                    out = torch.cat([out, pred_temp], dim=1)

            # save the sliding window on the image
            #cv2.imwrite('/home/xumin/yolov5/runs/slicing/' + 'sliding_window' + '.png', img_copy)

        pred = out
        #print('pred:',pred)
        #pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        #pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # nms

        #soft_nms
        #print('iou_thres:', iou_thres)
        #torch.set_printoptions(profile="full")
        #print('pred:',pred)
        #print('iou_thres:',iou_thres)
        pred = soft_nms(pred, conf_thres, iou_thres, multi_label=True)  # Soft DIoU-NMS
        print('pred_soft_nms:', out)
        #torch.set_printoptions(profile="full")
        #print('pred:',pred)
        #pred = soft_nms(pred, 0.1, 0.1, multi_label=True)  # Soft DIoU-NMS
        # Second-stage classifier (optional)
        if classify:
            pred = apply_classifier_aphid(out, modelc, im, original_src_copy)
            print('pred_cls:', pred)



        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        #pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        print('blurriness:', blurriness)



        for i, det in enumerate(pred):  # per image
            print('det:',det)
            # 提取置信度列（第五列）
            confidences = det[:, 4]
            # 计算平均值
            mean_confidence = torch.mean(confidences)
            mean_confidence_value = mean_confidence.item()
            print('mean_confidence_value:',mean_confidence_value)
            print('len(det):', len(det))

            # 存储计算结果的列表
            ratios_wh = []  # 宽高比
            edge_scores = []  # 边缘信息丰富度
            symmetry_scores = []  # 对称性分数
            texture_scores = []  # 纹理清晰度分数

            for box in det:
                # 提取边界框坐标
                x1, y1, x2, y2 = map(int, box[:4].cpu().numpy())
                width = x2 - x1
                height = y2 - y1

                # 1. 计算宽高比
                if height > 0:
                    ratio_wh = width / height
                    ratios_wh.append(ratio_wh)

                # 从原图像中截取预测框区域
                crop = original_src_copy[y1:y2, x1:x2]
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

                # 2. 边缘信息丰富度分析（使用Sobel算子检测边缘）
                sobelx = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
                sobel_edge_score = np.mean(np.hypot(sobelx, sobely))
                edge_scores.append(sobel_edge_score)

                # 3. 纹理特征分析（使用拉普拉斯算子检测纹理清晰度）
                laplacian = cv2.Laplacian(gray_crop, cv2.CV_64F)
                texture_score = np.var(laplacian)  # 计算纹理的方差作为清晰度
                texture_scores.append(texture_score)

                # 4. 对称性分析（水平翻转对比原图）
                flip_crop = cv2.flip(gray_crop, 1)  # 水平翻转图像
                symmetry_score = np.mean(np.abs(gray_crop - flip_crop))  # 对称性分数：原图与翻转图的差异
                symmetry_scores.append(symmetry_score)

            # 计算各项指标的平均值
            if len(ratios_wh) > 0:
                mean_ratio_wh = np.mean(ratios_wh)
                print(f'Mean Width-Height Ratio: {mean_ratio_wh}')

            if len(edge_scores) > 0:
                mean_edge_score = np.mean(edge_scores)
                print(f'Mean Edge Information Score: {mean_edge_score}')

            if len(texture_scores) > 0:
                mean_texture_score = np.mean(texture_scores)
                print(f'Mean Texture Clarity Score: {mean_texture_score}')

            if len(symmetry_scores) > 0:
                mean_symmetry_score = np.mean(symmetry_scores)
                print(f'Mean Symmetry Score: {mean_symmetry_score}')



            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            #print('type(im0):', type(im0))
            # print('pred_temp:',pred_temp)


            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            #print('type(im0)','type(im0)')
            #print('im0.size()',im0.shape)


            # number print on the image
            #count_number = "Aphids:{}".format(len(det))
            #cv2.putText(im0, count_number, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0,0), 15)



            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                #print('im.shape[2:]:', im.shape[2:])
                #print('im0.shape:',im0.shape)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #print('n:',n)

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #print('label:',label)
                        #label = None
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()

            # for calculate confusion_matrix #
            #xml_path = '/home/xumin/yolov5/dataset/stiring/5-7/GT/5_7_8_stiring_11.xml'
            xml_path = xml+ file_name+'.xml'
            print('xml_path:',xml_path)
            label = parse_xml(xml_path)
            #print('type(label):',type(label))
            #print('label', label)
            #print('label',label.size())
            #print('label.numel:',label.numel)
            det[:, 5] = 0
            pred = det
            #print('pred', pred)
            #print('pred', pred.size())
            #print('type(pred):', type(pred))
            #print('pred.numel:', pred.numel)
            matrix = np.zeros((nc + 1, nc + 1))
            Realiability = 0
            if (pred.numel() != 0) and (label.numel() != 0):
                confusion_matrix = ConfusionMatrix(nc=nc)
                confusion_matrix.process_batch(pred, label)
                # for single image
                confusion_matrix.plot_with_name(normalize=False, save_dir=save_dir, names=list(names.values()),image_name=file_name)
                #confusion_matrix.print()
                matrix=confusion_matrix.matrix_return()
                #print('type(matrix)',type(matrix))
                print('matrix:',matrix)
                # 提取 TP, FP, FN
                tp = matrix[0, 0]
                fp = matrix[0, 1]
                fn = matrix[1, 0]
                # 计算公式
                Realiability = tp / (tp + fp + fn)
                print(f"Realiability: {Realiability}")

                """
                # 计算各项指标的平均值
                if len(ratios_wh) > 0:
                    mean_ratio_wh = np.mean(ratios_wh)
                    print(f'Mean Width-Height Ratio: {mean_ratio_wh}')

                if len(edge_scores) > 0:
                    mean_edge_score = np.mean(edge_scores)
                    print(f'Mean Edge Information Score: {mean_edge_score}')

                if len(texture_scores) > 0:
                    mean_texture_score = np.mean(texture_scores)
                    print(f'Mean Texture Clarity Score: {mean_texture_score}')

                if len(symmetry_scores) > 0:
                    mean_symmetry_score = np.mean(symmetry_scores)
                    print(f'Mean Symmetry Score: {mean_symmetry_score}')
                """

                # for calculate confusion_matrix #
            cnbr.write(file_name + '.jpg' + ',' + str(mean_confidence_value) + ',' + str(len(det)) + ',' + str(
                blurriness) +',' + str(mean_ratio_wh) +',' + str(mean_edge_score) +',' +
                       str(mean_texture_score) +',' + str(mean_symmetry_score) +',' + str(Realiability) + '\n')

            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
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

                    cv2.imwrite(save_path, im0)

                    #count_number = "Aphids:{}".format(len(det))
                    #cv2.putText(im0, count_number, (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 15)
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
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        num = num+1

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
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
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--sliced_height', type=int, default=256, help='sliced_height')
    parser.add_argument('--sliced_width', type=int, default=256, help='sliced_width')
    parser.add_argument('--overlapped_height_ratio', type=float, default=0.1, help='overlapped_height_ratio')
    parser.add_argument('--overlapped_width_ratio', type=float, default=0.1, help='overlapped_height_ratio')
    parser.add_argument('--xml', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

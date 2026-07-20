#refer to https://github.com/obss/sahi
# https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb#scrollTo=9BnCswK5nbzj
# https://blog.csdn.net/djstavaV/article/details/120559239

import os
os.getcwd()

# arrange an instance segmentation model for test
from sahi.utils.yolov5 import (
    download_yolov5s6_model,
)

# import required functions, classes
from sahi.model import Yolov5DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image

# download YOLOV5S6 model to 'models/yolov5s6.pt'
yolov5_model_path = '/home/xumin/yolov5/runs/train/yolov5s-2-DCN2-low/weights/best.pt'
#download_yolov5s6_model(destination_path=yolov5_model_path)

# download test images into demo_data folder
#download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg', 'demo_data/small-vehicles1.jpeg')
#download_from_url('https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png', 'demo_data/terrain2.png')

detection_model = Yolov5DetectionModel(
    model_path=yolov5_model_path,
    confidence_threshold=0.25,
    device="cuda", # or 'cuda:0'
)

#without sliced prediction
#result = get_prediction("small-vehicles1.jpeg", detection_model)

result = get_sliced_prediction(
    "/home/xumin/yolov5/1_146_0.jpg",
    "/home/xumin/yolov5/sahi_slice/",
    detection_model,
    slice_height = 640, #256
    slice_width = 640, #256
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

#print('result',result)

result.export_visuals(export_dir="demo_data/")

object_prediction_list = result.object_prediction_list


Image("demo_data/result.png")
print('detection_number:',len(object_prediction_list))


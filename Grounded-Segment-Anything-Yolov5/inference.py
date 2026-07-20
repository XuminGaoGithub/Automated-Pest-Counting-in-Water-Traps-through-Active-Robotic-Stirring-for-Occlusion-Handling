import torch
from models.experimental import attempt_load

# Model
model_path = '/home/xumin/yolov5/yolov5s6.pt'
model=torch.load(model_path,map_location='cuda:0')
while(1):
# Images
    img = '/home/xumin/yolov5/demo_data/prediction_visual.png'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
    results = model(img)

# Results
    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    results.show()

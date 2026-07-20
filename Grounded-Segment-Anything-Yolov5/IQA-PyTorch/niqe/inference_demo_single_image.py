# conda activate yolov5
# pip install pyiqa
# pip install timm==0.6.7
# python inference_demo.py -m niqe --target /home/xumin/niqe_test/

import torch
from pyiqa import create_metric
import cv2
from PIL import Image

def niqe(image_path, device=None):
    """
    Calculate NIQE score for a given image.
    
    Args:
        image_path (str): Path to the input image.
        device (str, optional): Device to use for computation ('cuda' or 'cpu'). 
                               If None, will use GPU if available.
    
    Returns:
        float: NIQE score for the input image.
    """
    # Set device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create NIQE metric model (No-Reference metric)
    iqa_model = create_metric('niqe', metric_mode='NR', device=device)
    
    # Load image using OpenCV and convert to PIL Image (pyiqa expects PIL Image)
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Calculate NIQE score
    score = iqa_model(img_pil).cpu().item()
    
    return score

niqe_score = niqe('/home/newdrive/Phd/niqe/src/Aphid_10_frame_1.jpg')
print(f"NIQE score: {niqe_score:.4f}")

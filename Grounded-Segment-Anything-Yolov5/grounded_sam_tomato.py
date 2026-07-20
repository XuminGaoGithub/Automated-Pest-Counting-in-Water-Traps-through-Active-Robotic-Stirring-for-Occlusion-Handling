import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

#envpath = '/home/xumin/anaconda3/lib/python3.8/site-packages/cv2/qt/plugins/platforms'
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
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
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def calculate_overlap_area(box1, box2):
    # Calculate intersection area of two boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x1 < x2 and y1 < y2:
        intersection_area = (x2 - x1) * (y2 - y1)
    else:
        intersection_area = 0.0

    return intersection_area


def calculate_direction(tomato_centroid, leaf_centroid):
    x_diff = leaf_centroid[0] - tomato_centroid[0]
    y_diff = leaf_centroid[1] - tomato_centroid[1]

    direction = []

    # Determine vertical direction
    if y_diff < 0:
        direction.append('up')
    elif y_diff > 0:
        direction.append('down')

    # Determine horizontal direction
    if x_diff < 0:
        direction.append('left')
    elif x_diff > 0:
        direction.append('right')

    # Resolve combined directions
    if len(direction) == 2:
        if 'up' in direction and 'right' in direction:
            if abs(y_diff) > abs(x_diff):
                final_direction = 'up'
            else:
                final_direction = 'right'
        elif 'up' in direction and 'left' in direction:
            if abs(y_diff) > abs(x_diff):
                final_direction = 'up'
            else:
                final_direction = 'left'
        elif 'down' in direction and 'right' in direction:
            if abs(y_diff) > abs(x_diff):
                final_direction = 'down'
            else:
                final_direction = 'right'
        elif 'down' in direction and 'left' in direction:
            if abs(y_diff) > abs(x_diff):
                final_direction = 'down'
            else:
                final_direction = 'left'
    elif len(direction) == 1:
        final_direction = direction[0]
    else:
        final_direction = 'center'  # 如果没有方向变化（重合或几乎重合）

    return final_direction


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--sam_version", type=str, default="vit_b", required=False,
                        help="SAM ViT version: vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file")
    parser.add_argument("--use_sam_hq", action="store_true", help="using sam-hq for prediction")
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, default="tomato. leaf", help="text prompt")
    #parser.add_argument("--text_prompt", type=str, default="tomato", help="text prompt")

    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load image
    image_pil, image = load_image(args.input_image)

    # Load Grounding DINO model
    model = load_model(args.config, args.grounded_checkpoint, device=args.device)

    # Visualize raw image
    image_pil.save(os.path.join(args.output_dir, "raw_image.jpg"))

    # Run Grounding DINO model
    boxes_filt, pred_phrases, logits, tokenized = get_grounding_output(
        model, image, args.text_prompt, args.box_threshold, args.text_threshold, device=args.device
    )
    print('boxes_filt:',boxes_filt)
    print('pred_phrases:', pred_phrases)

    # Release GPU memory
    del model
    torch.cuda.empty_cache()

    # Initialize SAM
    if args.use_sam_hq:
        predictor = SamPredictor(
            sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint).to(args.device))
    else:
        predictor = SamPredictor(sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint).to(args.device))

    # Load and preprocess the image for SAM
    image = cv2.imread(args.input_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)


    # Transform boxes to match SAM coordinates
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    # Convert boxes to CPU
    boxes_filt = boxes_filt.cpu()

    # Transform boxes to SAM's coordinate system
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(args.device)

    # Predict masks using SAM
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )


    # Draw output image with masks and bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    print('yes')

    # Save the grounded SAM output image
    plt.savefig(os.path.join(args.output_dir, "grounded_sam_output.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)

    # Save mask data
    save_mask_data(args.output_dir, masks, boxes_filt, pred_phrases)

    # Find overlapping leafs that occlude tomato and calculate overlap areas
    tomato_index = None
    for idx, phrase in enumerate(pred_phrases):
        if 'tomato' in phrase:
            tomato_index = idx
            break

    leaf_index = None
    for idx, phrase in enumerate(pred_phrases):
        if 'leaf' in phrase:
            leaf_index = idx
            break

    if tomato_index is not None:
        tomato_box = boxes_filt[tomato_index]
        leaf_boxes = [box for idx, box in enumerate(boxes_filt) if 'leaf' in pred_phrases[idx] and idx != tomato_index]

        overlapping_leaf_boxes = []
        overlapping_areas = []
        for leaf_box in leaf_boxes:
            overlap_area = calculate_overlap_area(tomato_box, leaf_box)
            if overlap_area > 0:
                overlapping_leaf_boxes.append(leaf_box)
                overlapping_areas.append(overlap_area)
                print(f"Leaf overlaps tomato with area: {overlap_area}")

        """
        # Save mask showing overlapping and non-overlapping leaves with tomato
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for idx, box in enumerate(boxes_filt):
            if any(torch.equal(box, leaf_box) for leaf_box in overlapping_leaf_boxes):
                show_box(box.numpy(), plt.gca(), pred_phrases[idx])
            else:
                show_mask(masks[idx].cpu().numpy(), plt.gca(), random_color=False)
        plt.axis('off')
        plt.savefig(os.path.join(args.output_dir, "overlap_mask.jpg"), bbox_inches="tight", dpi=300, pad_inches=0.0)
        """


        
        # Find the largest overlapping leaf
        if overlapping_leaf_boxes:
            largest_overlap_idx = np.argmax(overlapping_areas)
            largest_overlap_leaf_box = overlapping_leaf_boxes[largest_overlap_idx]
            #print('masks:',masks)


            #print('pred_phrases:',pred_phrases)
            # Ensure tomato_index and largest_overlap_leaf_index are different
            while tomato_index == largest_overlap_idx:
                tomato_index = (tomato_index + 1) % len(pred_phrases)

            # Calculate centroid of tomato using SAM mask
            tomato_mask = masks[tomato_index].cpu().numpy()
            #print('tomato_index:',tomato_index)
            # Flatten the mask tensor
            tomato_mask_flat = tomato_mask.squeeze().reshape(-1)
            # Find non-zero elements
            nonzero_indices = np.where(tomato_mask_flat > 0)
            # Convert flat indices to coordinates
            y_coords, x_coords = np.unravel_index(nonzero_indices, tomato_mask.shape[1:])
            # Calculate centroid only if there are non-zero elements
            if len(y_coords) > 0:
                tomato_centroid = np.array([np.mean(x_coords), np.mean(y_coords)])
                print(f"Tomato centroid (SAM): {tomato_centroid}")
            else:
                print("No non-zero elements found in tomato_mask.")


            # Calculate centroid of the largest overlapping leaf using SAM mask
            largest_overlap_leaf_mask = masks[leaf_boxes.index(largest_overlap_leaf_box)].cpu().numpy()
            #print('largest_overlap_leaf_box:',largest_overlap_leaf_box)
            #print('leaf_boxes.index(largest_overlap_leaf_box):', leaf_boxes.index(largest_overlap_leaf_box))
            largest_overlap_leaf_mask = masks[leaf_boxes.index(largest_overlap_leaf_box)].cpu().numpy()

            # Flatten the mask tensor
            largest_overlap_leaf_mask_flat = largest_overlap_leaf_mask.squeeze().reshape(-1)
            # Find non-zero elements
            nonzero_indices = np.where(largest_overlap_leaf_mask_flat)
            # Convert flat indices to coordinates
            y_coords, x_coords = np.unravel_index(nonzero_indices, largest_overlap_leaf_mask.shape[1:])
            # Calculate centroid only if there are non-zero elements
            if len(y_coords) > 0:
                largest_overlap_leaf_centroid = np.array([np.mean(x_coords), np.mean(y_coords)])
                print(f"Largest overlapping leaf centroid (SAM): {largest_overlap_leaf_centroid}")
            else:
                print("No non-zero elements found in largest_overlap_leaf_mask.")


            # Convert centroids to original image coordinates
            print('W, image.shape[1], H, image.shape[0]:',W, image.shape[1], H, image.shape[0])
            tomato_centroid_orig = tomato_centroid * [W / image.shape[1], H / image.shape[0]]
            largest_overlap_leaf_centroid_orig = largest_overlap_leaf_centroid * [W / image.shape[1], H / image.shape[0]]
            print(f"Tomato centroid (original image): {tomato_centroid_orig}")
            print(f"Largest overlapping leaf centroid (original image): {largest_overlap_leaf_centroid_orig}")

            # 计算方向
            final_direction = calculate_direction(tomato_centroid_orig, largest_overlap_leaf_centroid_orig)
            # 打印方向
            print(f"The direction of the leaf relative to the tomato is: {final_direction}")



            # Save the coordinates to tomato.txt
            with open('tomato.txt', 'w') as file:
                #file.write(f"{tomato_centroid_orig[0]},{tomato_centroid_orig[1]}\n")
                file.write(f"{largest_overlap_leaf_centroid_orig[0]},{largest_overlap_leaf_centroid_orig[1]}\n")
                file.write(f"{final_direction}")

            # Draw centroids on the original image
            image_pil_orig = Image.open(args.input_image).convert("RGB")
            draw = ImageDraw.Draw(image_pil_orig)
            draw.ellipse((tomato_centroid_orig[0] - 5, tomato_centroid_orig[1] - 5, tomato_centroid_orig[0] + 5,
                          tomato_centroid_orig[1] + 5), fill='blue')
            draw.ellipse((largest_overlap_leaf_centroid_orig[0] - 5, largest_overlap_leaf_centroid_orig[1] - 5,
                          largest_overlap_leaf_centroid_orig[0] + 5, largest_overlap_leaf_centroid_orig[1] + 5),
                         fill='red')

            # Save image with centroids
            image_pil_orig.save(os.path.join(args.output_dir, 'overlap_centroids_original.jpg'))


            # Plot centroids on the grounded SAM output image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for idx, box in enumerate(boxes_filt):
                if any(torch.equal(box, leaf_box) for leaf_box in overlapping_leaf_boxes):
                    show_box(box.numpy(), plt.gca(), pred_phrases[idx])
            plt.scatter(largest_overlap_leaf_centroid[0], largest_overlap_leaf_centroid[1], c='red', s=100, marker='o')
            plt.scatter(tomato_centroid[0], tomato_centroid[1], c='blue', s=100, marker='o')
            plt.axis('off')

            # Save image with centroids
            plt.savefig(os.path.join(args.output_dir, 'overlap_centroids.jpg'), bbox_inches="tight", dpi=300,
                        pad_inches=0.0)

    plt.show()

"""
import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
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
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_b", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_image", type=str, required=True, help="path to image file")
    #parser.add_argument("--text_prompt", type=str, default="tomato. leaf", required=True, help="text prompt")

    parser.add_argument("--text_prompt", type=str, default="tomato. leaf", help="text prompt")

    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    image_path = args.input_image
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # load image
    image_pil, image = load_image(image_path)
    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    # visualize raw image
    image_pil.save(os.path.join(output_dir, "raw_image.jpg"))

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, device=device
    )

    del model
    torch.cuda.empty_cache()  # 释放 GPU 缓存
    #model.cpu()

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

    masks, _, _ = predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(boxes_filt, pred_phrases):
        show_box(box.numpy(), plt.gca(), label)

    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, "grounded_sam_output.jpg"),
        bbox_inches="tight", dpi=300, pad_inches=0.0
    )

    save_mask_data(output_dir, masks, boxes_filt, pred_phrases)
"""
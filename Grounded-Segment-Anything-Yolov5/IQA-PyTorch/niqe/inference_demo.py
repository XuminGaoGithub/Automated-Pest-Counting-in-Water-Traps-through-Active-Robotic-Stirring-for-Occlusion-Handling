# conda activate yolov5
# pip install pyiqa
# pip install timm==0.6.7
# python inference_demo.py -m niqe --target /home/xumin/niqe_test/

import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import csv
from time import time
from PIL import Image, ImageDraw, ImageFont
import torch

def main():
    """Inference demo for pyiqa."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default=None, help='input image/folder path.')
    parser.add_argument('-r', '--ref', type=str, default=None, help='reference image/folder path if needed.')
    parser.add_argument('--device', type=str, default=None, help='reference image/folder path if needed.')
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='FR',
        help='metric mode Full Reference or No Reference. options: FR|NR.')
    parser.add_argument('-m', '--metric_name', type=str, default='PSNR', help='IQA metric name, case sensitive.')
    parser.add_argument('--save_file', type=str, default=None, help='path to save results.')

    # Add a --verbose flag
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',  # This makes it a flag (True when used, False otherwise)
        help='Enable verbose output'
    )

    args = parser.parse_args()

    metric_name = args.metric_name.lower()

    # Create result directory if it doesn't exist
    result_dir = os.path.join('./result', metric_name)  # Save to ./result/metric_name
    os.makedirs(result_dir, exist_ok=True)

    # Set up IQA model
    iqa_model = create_metric(metric_name, metric_mode=args.metric_mode, device=args.device)
    metric_mode = iqa_model.metric_mode

    if os.path.isfile(args.target):
        input_paths = [args.target]
        if args.ref is not None:
            ref_paths = [args.ref]
    else:
        input_paths = sorted(glob.glob(os.path.join(args.target, '*')))
        if args.ref is not None:
            ref_paths = sorted(glob.glob(os.path.join(args.ref, '*')))

    # Open CSV file for writing
    csv_file_path = os.path.join(result_dir, f'{metric_name}.csv')
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image Name', 'Score'])

        avg_score = 0
        test_img_num = len(input_paths)
        if not 'fid' in metric_name:
            pbar = tqdm(total=test_img_num, unit='image')
            for idx, img_path in enumerate(input_paths):
                img_name = os.path.basename(img_path)
                if metric_mode == 'FR':
                    ref_img_path = ref_paths[idx]
                else:
                    ref_img_path = None

                start_time = time()
                score = iqa_model(img_path, ref_img_path).cpu().item()
                end_time = time()
                avg_score += score
                pbar.update(1)
                pbar.set_description(f'{metric_name} of {img_name}: {score:.4f}')
                pbar.write(f'{metric_name} of {img_name}: {score:.4f}\tTime: {end_time - start_time:.2f}s')

                # Write to CSV
                csvwriter.writerow([img_name, f'{score:.4f}'])

                # Load image and draw score on it
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=60)
                except IOError:
                    font = ImageFont.load_default()  # Fallback to default font

                # Display the score on the image (formatted to 4 decimal places)
                draw.text((10, 10), f"{metric_name}_score: {score:.4f}", fill="red", font=font)

                # Save the modified image to the result directory
                result_img_path = os.path.join(result_dir, img_name)
                img.save(result_img_path)

            pbar.close()
            avg_score /= test_img_num
        else:
            assert os.path.isdir(args.target) and os.path.isdir(args.ref), 'input path must be a folder for FID.'
            avg_score = iqa_model(args.target, args.ref)

    if args.verbose and torch.cuda.is_available():
        print(torch.cuda.memory_summary())

    msg = f'Average {metric_name} score of {args.target} with {test_img_num} images is: {avg_score:.4f}'
    print(msg)

    print(f'Done! Results are in {result_dir}.')

if __name__ == '__main__':
    main()

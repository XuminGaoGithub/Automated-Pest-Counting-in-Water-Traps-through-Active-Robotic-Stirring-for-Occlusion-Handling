
# <p align="center"> Automated Pest Counting in Water Traps through Active Robotic Stirring for Occlusion Handling </p>



<p align="center">
  <img src="https://github.com/XuminGaoGithub/Counting-with-Confidence-Accurate-Pest-Monitoring-in-Water-Traps/blob/main/Counting%20with%20Confidence/overview_C5.jpg" width="1000" height="1000"" />
</p>

<p align="center">
Overview of our proposed method
</p>
<br/>


## Overview
Existing image-based pest counting methods rely on single static images and often produce inaccurate results under occlusion. To address this issue, this paper proposes an automated pest counting method in water traps through active robotic stirring. First, an automated robotic arm-based stirring system is developed to redistribute pests and reveal occluded individuals for counting. Then, the effects of different stirring patterns on pest counting performance are investigated. Six stirring patterns are designed and evaluated across different pest density scenarios to identify the optimal one. Finally, a heuristic counting confidence-driven closed-loop control system is proposed for adaptive-speed robotic stirring, adjusting the stirring speed based on the average change rate of counting confidence between consecutive frames. Experimental results show that the four circles is the optimal stirring pattern, achieving the lowest overall mean absolute counting error of 4.384 and the highest overall mean counting confidence of 0.721. Compared with constant-speed stirring, adaptive-speed stirring reduces task execution time by up to 44.7% and achieves more stable performance across different pest density scenarios. Moreover, the proposed pest counting method reduces the mean absolute counting error by up to 3.428 compared to the single static image counting method under high-density scenarios where occlusion is severe.



## Different stirring patterns

As space limitation and file size limitation, datasets and some of models are uplaoded to OneDrive. Please download them by the links provided.

***1. Set up Grounded-Segment-Anything-Yolov5 envs***

Please refer https://github.com/XuminGaoGithub/Automatic_aphid_counting___2023/tree/main/Automatic_aphid_counting to set up Yolov5 envs and refer https://github.com/idea-research/grounded-segment-anything to set up Grounded-Segment-Anything envs.


***2. Franka_arm ROS2 envs***

Please refer https://github.com/LCAS/franka_arm_ros2 to set up Franka_arm ROS2 envs

***3. Dataset collection***

add ws_active_perception and ws_robots to franka_arm_ros2, build both packages. 

franka@franka-PC-BX19152:~/ws_robots$ source ~/ws_robots/install/setup.bash
franka@franka-PC-BX19152:~/ws_robots$ ros2 launch franka_moveit_config moveit_real_arm.launch.py robot_ip:=172.16.0.2

franka@franka-PC-BX19152:~/ws_active_perception$ source ~/ws_active_perception/install/setup.bash
cd ~/ws_active_perception

# Use the code annotated with /*1-6*/ in src/moveit_test.cpp to make robot arm execute different stirring patterns

colcon build --packages-select moveit2_commander_recorder 

source install/setup.bash

franka@franka-PC-BX19152:~/ws_active_perception$ ros2 launch moveit2_commander_recorder moveit_test.launch.py

ros2 launch moveit2_commander_recorder moveit_test.launch.py shape:=round video_name:=low_round_5 #Running example, can set different patterns, different density and rename video, therefore collect all of videos with different stirring patterns.


Please find the collected videos from OneDrive (Link: https://universityoflincoln-my.sharepoint.com/:f:/g/personal/25766099_students_lincoln_ac_uk/IgA26xX3DABHTo3NX2pM3lblAUgJDKPea57kkTtxX0NY7M4?e=AcZvii)

***4. Data processing and analysis***

(1) Get detection result .txt files and [result.write(file_name + '.jpg' + ',' + str(mean_confidence_value) + ',' + str(predict_number) + ',' + str(
                blurriness) + ',' + str(entropy) + ',' + str(niqe) + ',' + str(score) + '\n')] into ./simulation_output/score/result_records.txt

# Set the path to the folder containing multiple .avi video files
video_folder = "..." #path of the collected videos

#txt without BBOX confidence
python counting_confidence_detect_ground-sam_v9.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /home/xumin/Open-GroundingDino/weights/groundingdino_swint_ogc.pth --sam_hq_checkpoint /home/newdrive/summerschool/Grounded-Segment-Anything/sam_hq_vit_b.pth --use_sam_hq  --input_dir ./counting_confidence/src --output_dir "./simulation_output/image_processing" --box_threshold 0.5 --text_threshold 0.25 --device "cuda" --weights /home/newdrive/Phd/Modification_yolov5/runs/train/voc2060_yolov5s-ODConv-cotnet2/weights/best.pt --device_yolo 0 --save-txt --source ./counting_confidence/yellow_pan


#txt with BBOX confidence
python counting_confidence_detect_ground-sam_v9.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /home/xumin/Open-GroundingDino/weights/groundingdino_swint_ogc.pth --sam_hq_checkpoint /home/newdrive/summerschool/Grounded-Segment-Anything/sam_hq_vit_b.pth --use_sam_hq  --input_dir ./counting_confidence/src --output_dir "./simulation_output/image_processing" --box_threshold 0.5 --text_threshold 0.25 --device "cuda" --weights /home/newdrive/Phd/Modification_yolov5/runs/train/voc2060_yolov5s-ODConv-cotnet2/weights/best.pt --device_yolo 0 --save-txt --save-conf --source ./counting_confidence/yellow_pan

(2) Ground Truth labels

Get gt labels by annotations (./Annotations) 

(3) Confusionmatrix, visualisation_TPFPFN, Saving TP,FP,FN, Realiability = tp / (tp + fp + fn)， gt_image (Ground truth insect number in the image)， gt_real（Ground truth insect number in the yellow water pan）into counting_result_information_xxx_density_shapes.txt

#convert annotaions .xml to yolo format .txt
 
copy ./Annotations & ./yellow_pan_xxx_density_shapes (rename as ./JPEGImages) to ./dataset/voc20xx, copy and set up /voc20xx/ImageSets

python txt_2.py #generate /voc20xx/ImageSets/Main/test.txt  train.txt  trainval.txt  val.txt

python voc_label_20xx.py # to get yolo format labels ./dataset/voc20xx/labels


#run python vis_TPFPFN.py

#modify gt_real = 6 # 12,18 gt number of aphids in the yellow pan
#modify paths：
img_path = './dataset/voc20xx/JPEGImages/'
Annotations_path = './dataset/voc20xx/Annotations' #gt_Annotations_VOC_format
label_path = './dataset/voc20xx/labels/' #gt_Annotations_yolo_format
predict_path = './runs/detect/counting_confidence_xxx_density_shapes_without_conf/pred_labels' #pred_labels
save_path = './runs/detect/counting_confidence_xxx_density_shapes_without_conf/vis_TPFPFN/'

python vis_TPFPFN.py 

# If do not want the bounding box confidence score,
# uncomment the line below and set confidence to None.
# confidence = pred[max_iou_idx][5]  # Get the confidence score
confidence = None

(4) calculate MAE = (GT_real - TP) and Average Counting Confidence for each of stirring patterns

#copy result_records.txt (rename as result_records_xxx_density_shapes.txt) and counting_result_information_xxx_density_shapes.txt to ./analysis/

python sorting.py 

python analysis_total.py  #calculate MAE = (GT_real - TP) and Average Counting Confidence for each of stirring patterns

# please modify code according to different densities
records_file = "result_records_low_density_shapes.txt"
counting_file = "counting_result_information_low_density_shapes_sorted.txt"

match = re.match(r"low_(\w+)_(\d+)_frame_(\d+)\.jpg", name)
#match = re.match(r"med_(\w+)_(\d+)_frame_(\d+)\.jpg", name)
#match = re.match(r"high_(\w+)_(\d+)_frame_(\d+)\.jpg", name)



## Adaptive-speed stirring

***1. ROS2:Use the code annotated with /*2-4*/ in src/moveit_test.cpp to make robot arm execute adaptive/constant-speed stirring***

colcon build --packages-select moveit2_commander_recorder 

source install/setup.bash

franka@franka-PC-BX19152:~/ws_active_perception$ ros2 launch moveit2_commander_recorder moveit_test.launch.py



***2.Perception***

python counting_confidence_detect_ground-sam_v10_20250903.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint /home/xumin/Open-GroundingDino/weights/groundingdino_swint_ogc.pth --sam_hq_checkpoint /home/newdrive/summerschool/Grounded-Segment-Anything/sam_hq_vit_b.pth --use_sam_hq  --input_dir ./counting_confidence/src --output_dir "./simulation_output/image_processing" --box_threshold 0.5 --text_threshold 0.25 --device "cuda" --weights /home/newdrive/Phd/Modification_yolov5/runs/train/voc2060_yolov5s-ODConv-cotnet2/weights/best.pt --device_yolo 0 --save-txt --source ./counting_confidence/yellow_pan

#Results will be saved to ./counting_confidence, ./runs/detect/counting_confidence, and ./simulation_output, and progress information will also be printed to the console.













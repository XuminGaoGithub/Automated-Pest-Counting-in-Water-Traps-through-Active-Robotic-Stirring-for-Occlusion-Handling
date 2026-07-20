#python train.py --img 640 --batch 2 --epoch 2000 --data data/aphid_voc.yaml --cfg models/yolov5_RepLKNet.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 500
#python train.py --img 640 --batch 2 --epoch 2000 --data data/aphid_voc.yaml --cfg models/yolov5s-BoTNet.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 500
#python train.py --img 640 --batch 2 --epoch 2000 --data data/aphid_voc.yaml --cfg models/yolov5s_cotnet.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 500
#python train.py --img 640 --batch 2 --epoch 2000 --data data/aphid_voc.yaml --cfg models/yolov5_swin_transfomrer.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 500
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s_cotnet2.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-BoTNet2.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_ConvNeXt_tiny.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-ODConv.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-DCN-V3-2.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-EVCBlock.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 100
#python train.py --img 256 --batch 2 --epoch 600 --data data/aphid_voc_2019.yaml --cfg models/yolov5s.yaml --weights weights/yolov5s.pt --device '0' --patience 0 --save-period 100

#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_EfficientFormerV2.yaml  --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_fasternet.yaml --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_inceptionnext.yaml --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_swin_transformer_tiny.yaml --device '0' --patience 0 --save-period 100 #nan without pre-weight
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_efficientViT.yaml --device '0' --patience 0 --save-period 100 #nan 


#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_EfficientFormerV2.yaml  --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_fasternet.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_fasternet-l.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt


#python train.py --img 512 --batch 4 --epoch 300 --data data/aphid_voc_2020.yaml --cfg models/yolov5s-final.yaml --device '0' --patience 0 --save-period 100

#python train.py --img 640 --batch 4 --epoch 600 --data data/aphid_voc_2032.yaml --cfg models/yolov5s-final.yaml --device '0' --patience 0 --save-period 100

#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_EfficientFormerV2-l.yaml  --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-transformer.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-transformer-2.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 2 --data data/aphid_voc.yaml --cfg models/yolov5_swin_transformer_tiny.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt #nan
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5_efficientViT.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt #nan

#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-CA-1.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-CA-2.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-CA-3.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt

#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s_C3_Res2Block.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-CBAM.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-SEAttention.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s-SimAM.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt

#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg models/yolov5s.yaml --device '0' --patience 0 --save-period 100 --weights weights/yolov5s.pt #nwd

#python train_ota.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg yolov5s.yaml --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg yolov5_GFPN.yaml --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg yolov5s_ContextA.yaml --device '0' --patience 0 --save-period 100
#python train.py --img 640 --batch 2 --epoch 600 --data data/aphid_voc.yaml --cfg yolov5s_cam.yaml --device '0' --patience 0 --save-period 100



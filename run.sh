#!/usr/bin/env bash
nohup python train_hdn.py \
   --gpu=3  --load_RCNN=1 --base_model=vgg  --mps_feature_len=2048  --lr=0.001  --dataset_option=small  --MPS_iter=1 --log_interval=1000 --step_size=6 \
 --RCNN_model=./output/detection/Faster_RCNN_vgg_12epoch_2048_best.h5 \
> nohup/hdn_rcnn_vgg_1iter_small_onlyfaster_2048.out 2>&1 &

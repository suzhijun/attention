#!/usr/bin/env bash
nohup python train_hdn.py \
   --gpu=2  --load_RCNN=1 --base_model=vgg  --mps_feature_len=2048  --lr=0.001  --dataset_option=all  --MPS_iter=2 --log_interval=1000 --step_size=3 --max_epoch=12  --RCNN_model=./output/detection/Faster_RCNN_vgg_12epoch_2048_best.h5 \
> nohup/hdn_rcnn_vgg_2iter_all_2048.out 2>&1 &

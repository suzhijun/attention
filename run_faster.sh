#!/usr/bin/env bash
nohup python train_faster_rcnn.py \
   --gpu=0  --step_size=3  --max_epoch=12 --lr=0.001  --dataset_option=normal  --base_model=resnet101  \
   --clip_gradient --model_name=Faster_RCNN_resnet101_12epoch_bn_2048 --log_interval=1000 \
>nohup/rcnn_normal_resnet101_bn_12epoch_2048.out 2>&1 &

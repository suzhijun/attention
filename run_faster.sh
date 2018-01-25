#!/usr/bin/env bash
python train_faster_rcnn.py \
   --gpu=0  --lr=0.01  --dataset_option=small  --step_size=2  --max_epoch=8 --model_name=Faster_RCNN_resnet50

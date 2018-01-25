#!/usr/bin/env bash
python train_faster_rcnn.py \
   --gpu=0  --step_size=2  --max_epoch=8 --lr=0.01  --dataset_option=small  --model=resnet  --clip_gradient --model_name=Faster_RCNN_resnet50 --log_interval=500

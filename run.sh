#!/usr/bin/env bash
python train_hdn.py \
   --gpu=0  --load_RCNN=1  --lr=0.001  --dataset_option=small  --MPS_iter=2 |tee ./log/hdn_label.out

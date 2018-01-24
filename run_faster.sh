#!/usr/bin/env bash
nohup python train_hdn.py \
   --gpu=0  --lr=0.01  --dataset_option=small  --MPS_iter=3 \
>nohup/hdn_label_3iter.out 2>&1 &

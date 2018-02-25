#!/usr/bin/env bash

python train_hdn.py \
    --gpu=0 --resume_model --HDN_model=./output/HDN_1_iters_all_vgg_2048_kmeans_RCNN_SGD_best.h5 \
    --dataset_option=all  --mps_feature_len=2048 --MPS_iter=1 --evaluate

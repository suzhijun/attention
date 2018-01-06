#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_hdn.py \
    --resume_training --resume_model ./output/HDN/HDN_2_iters_alt_normal_no_caption_SGD_best.h5 \
    --dataset_option=normal  --MPS_iter=2 \
    --disable_language_model --evaluate
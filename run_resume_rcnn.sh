nohup python train_hdn.py \
   --gpu=0  --step_size=2  --max_epoch=10 --lr=0.01  --dataset_option=small  --base_model=resnet101 --model_name=HDN_RCNN_small_resnet101_notrain_largelr --MPS_iter=1 --log_interval=1000 \
  --load_RCNN=True  --RCNN_model=./output/detection/Faster_RCNN_small_resnet101_16epoch_epoch_15.h5 \
>nohup/hdn_notrain_largelr_rcnn_small_resnet101_1iter_10epoch.out 2>&1 &

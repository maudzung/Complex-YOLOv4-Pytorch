#!/usr/bin/env bash
python train.py \
  --saved_fn 'complex_yolov4' \
  --arch 'darknet' \
  --cfgfile ./config/complex_yolov4.cfg \
  --batch_size 4 \
  --num_workers 4 \
  --no-val \
  --gpu_idx 0 \
  --lr_type 'cosin' \
  --lr 0.001 \
  --num_epochs 300 \
  --tensorboard_freq 50 \
  --checkpoint_freq 5 \
  --multiscale_training

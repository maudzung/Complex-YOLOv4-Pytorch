#!/usr/bin/env bash
python train.py \
  --saved_fn 'complex_yolov4_20200708' \
  --arch 'darknet' \
  --cfgfile ./config/complex_yolov4.cfg \
  --batch_size 4 \
  --num_workers 4 \
  --no-val \
  --print_freq 50 \
  --tensorboard_freq 20 \
  --checkpoint_freq 2 \
  --gpu_idx 1 \
  --lr 0.0025 \
  --burn_in 50 \
  --steps 1500 4000 \
  --multiscale_training

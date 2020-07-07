#!/usr/bin/env bash
python train.py \
  --saved_fn 'complex_yolov4_20200707' \
  --arch 'darknet' \
  --cfgfile ./config/complex_yolov4.cfg \
  --batch_size 4 \
  --num_workers 4 \
  --no-val \
  --print_freq 50 \
  --tensorboard_freq 20 \
  --checkpoint_freq 5 \
  --gpu_idx 0 \
  --burn_in 125 \
  --steps 500 1000
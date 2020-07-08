#!/usr/bin/env bash
python train.py \
  --saved_fn 'complex_yolov4' \
  --arch 'darknet' \
  --cfgfile ./config/complex_yolov4.cfg \
  --batch_size 4 \
  --num_workers 4 \
  --no-val \
  --gpu_idx 0 \
  --lr 0.0025 \
  --burn_in 50 \
  --steps 1500 4000 \
  --multiscale_training

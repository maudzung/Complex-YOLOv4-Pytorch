#!/usr/bin/env bash
python train.py \
  --saved_fn 'complex_yolov4' \
  --arch 'darknet' \
  --cfgfile ./config/complex_yolov4.cfg \
  --batch_size 4 \
  --num_workers 4 \
  --no-val \
  --gpu_idx 1 \
  --lr 0.0025 \
  --num_epochs 150 \
  --burn_in 100 \
  --steps 2000 5000 \
  --multiscale_training

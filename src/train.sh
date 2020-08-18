#!/usr/bin/env bash
python train.py \
  --saved_fn 'complex_yolov4' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/complex_yolov4.cfg \
  --batch_size 4 \
  --num_workers 4 \
  --no-val \
  --gpu_idx 0
#!/usr/bin/env bash
python train.py \
  --saved_fn 'complex_yolov4_20200707' \
  --arch 'darknet' \
  --cfgfile ./config/complex_yolov4.cfg \
  --batch_size 16 \
  --num_workers 8 \
  --no-val \
  --print_freq 50 \
  --checkpoint_freq 5 \
  --world-size 1 \
  --rank 0 \
  --dist-backend 'nccl' \
  --multiprocessing-distributed
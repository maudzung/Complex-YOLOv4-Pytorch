#!/usr/bin/env bash
python test.py \
  --saved_fn 'complex_yolov4' \
  --arch 'darknet' \
  --cfgfile ./config/complex_yolov4.cfg \
  --batch_size 1 \
  --num_workers 1 \
  --gpu_idx 0 \
  --pretrained_path ../checkpoints/complex_yolov4/Model_complex_yolov4_epoch_300.pth \
  --img_size 608 \
  --conf_thresh 0.5 \
  --nms_thresh 0.5 \
  --show_image \
  --save_test_output \
  --output_format 'image'

"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.08
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: Testing script
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes
from utils.misc import time_synchronized
from utils.prediction_utils import predictions_to_kitti_format
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev


def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolo', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.7,
                        help='the threshold for conf')

    parser.add_argument('--video_path', type=str, default=None, metavar='PATH',
                        help='the path of the video that needs to demo')
    parser.add_argument('--output_format', type=str, default='text', metavar='PATH',
                        help='the type of the demo output')
    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_demo_output', action='store_true',
                        help='If true, the image of demonstration phase will be saved')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.saved_fn)
    configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
    make_folder(configs.results_dir)

    return configs


if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing

    model = create_model(configs)
    model.print_network()
    assert os.path.isfile(configs.pretrained_path)
    model.load_weights(configs.pretrained_path)
    # model.load_state_dict(torch.load(configs.pretrained_path))

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)

    model.eval()

    test_dataloader = create_test_dataloader(configs)
    with torch.no_grad():
        for batch_idx, (img_paths, imgs_bev) in enumerate(test_dataloader):
            print('batch_idx: {}'.format(batch_idx))
            input_imgs = imgs_bev.to(device=configs.device).float()
            t1 = time_synchronized()
            detections = model(input_imgs)
            t2 = time_synchronized()
            print('outputs size: {}'.format(detections))
            detections = post_processing(detections, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            img_detections = []  # Stores detections for each image index
            img_detections.extend(detections)

            img_bev = imgs_bev.squeeze() * 255
            img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
            img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
            for detections in img_detections:
                if detections is None:
                    continue
                # Rescale boxes to original image
                detections = np.array(detections)
                detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
                for x, y, w, l, im, re, cls_conf, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw rotated box
                    kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

            img_rgb = cv2.imread(img_paths[0])
            calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, configs.img_size)
            img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

            img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

            out_img = merge_rgb_to_bev(img_rgb, img_bev, output_width=608)
            cv2.imshow('test-img', out_img)

            print('Done testing the {} sample, time: {}ms'.format(batch_idx, (t2 - t1) * 1000))

            if cv2.waitKey(0) & 0xFF == 27:
                break

"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for creating the dataloader for training/validation/test phase
"""

import sys

import torch
from torch.utils.data import DataLoader

sys.path.append('../')

from data_process.kitti_dataset import KittiDataset


def create_train_val_dataloader(configs):
    """Create dataloader for training and validate"""
    train_dataset = KittiDataset(configs.dataset_dir, split='train', mode='train', data_aug=True,
                                 multiscale=configs.multiscale_training, num_samples=configs.num_samples)
    train_sampler = None
    if configs.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=(train_sampler is None),
                                  pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=train_sampler,
                                  collate_fn=train_dataset.collate_fn)

    val_sampler = None
    val_dataset = KittiDataset(configs.dataset_dir, split='val', mode='val', data_aug=False, multiscale=False,
                               num_samples=configs.num_samples)
    if configs.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False,
                                pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=val_sampler,
                                collate_fn=val_dataset.collate_fn)

    return train_dataloader, val_dataloader, train_sampler


def create_test_dataloader(configs):
    """Create dataloader for testing phase"""

    test_dataset = KittiDataset(configs.dataset_dir, split='test', mode='test', data_aug=False,
                                multiscale=False, num_samples=configs.num_samples)
    test_sampler = None
    if configs.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size, shuffle=False,
                                 pin_memory=configs.pin_memory, num_workers=configs.num_workers, sampler=test_sampler,
                                 collate_fn=test_dataset.collate_fn)

    return test_dataloader


if __name__ == '__main__':
    import cv2
    import numpy as np

    from config.config import parse_configs
    import data_process.kitti_bev_utils as bev_utils
    from data_process import kitti_data_utils
    from inference.prediction_utils import invert_target
    from utils.visualization_utils import show_image_with_boxes

    configs = parse_configs()
    configs.distributed = False  # For testing
    train_dataloader, val_dataloader, train_sampler = create_train_val_dataloader(configs)
    print('len train_dataloader: {}, val_dataloader: {}'.format(len(train_dataloader), len(val_dataloader)))

    img_size = configs.BEV_WIDTH

    for batch_i, (img_files, imgs, targets) in enumerate(val_dataloader):
        img_file = img_files[0]
        img_rgb = cv2.imread(img_file)
        calib = kitti_data_utils.Calibration(img_file.replace(".png", ".txt").replace("image_2", "calib"))
        objects_pred = invert_target(targets[:, 1:], calib, img_rgb.shape, RGB_Map=None)
        img_rgb = show_image_with_boxes(img_rgb, objects_pred, calib, False)

        # Rescale target
        targets[:, 2:6] *= img_size
        # Get yaw angle
        targets[:, 6] = torch.atan2(targets[:, 6], targets[:, 7])

        img = imgs.squeeze() * 255
        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        img_display = np.zeros((img_size, img_size, 3), np.uint8)
        img_display[...] = img[...]

        for c, x, y, w, l, yaw in targets[:, 1:7].numpy():
            # Draw rotated box
            bev_utils.drawRotatedBox(img_display, x, y, w, l, yaw, configs.colors[int(c)])

        cv2.imshow('img-kitti-bev', img_display)
        cv2.imshow('img_rgb', img_rgb)

        if cv2.waitKey(0) & 0xff == 27:
            break

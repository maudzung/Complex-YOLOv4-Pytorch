"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
"""

import sys
import os
from glob import glob
import random

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2

sys.path.append('../')

from data_process import transformation, kitti_bev_utils, kitti_data_utils


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class KittiDataset(Dataset):
    def __init__(self, configs, split='train', mode='TRAIN', data_aug=True, multiscale=False):
        self.configs = configs
        self.dataset_dir = configs.dataset_dir
        self.split = split
        self.multiscale = multiscale
        self.data_aug = data_aug
        self.img_size = configs.BEV_WIDTH
        self.max_objects = 100
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        is_test = self.split == 'test'
        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode

        if is_test:
            self.lidar_dir = os.path.join(self.dataset_dir, 'testing', "velodyne")
            self.image_dir = os.path.join(self.dataset_dir, 'testing', "image_2")
            self.calib_dir = os.path.join(self.dataset_dir, 'testing', "calib")
            self.label_dir = os.path.join(self.dataset_dir, 'testing', "label_2")
            self.lidar_paths = sorted(glob(os.path.join(self.lidar_dir, '*.bin')))
            self.image_idx_list = [os.path.basename(path)[:-4] for path in self.lidar_paths]
            print(self.image_idx_list[0])
        else:
            self.lidar_dir = os.path.join(self.dataset_dir, 'training', "velodyne")
            self.image_dir = os.path.join(self.dataset_dir, 'training', "image_2")
            self.calib_dir = os.path.join(self.dataset_dir, 'training', "calib")
            self.label_dir = os.path.join(self.dataset_dir, 'training', "label_2")
            split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(split))
            self.image_idx_list = [x.strip() for x in open(split_txt_path).readlines()]
        self.sample_id_list = []

        if mode == 'train':
            self.preprocess_training_data()
        else:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

        if configs.num_samples is not None:
            self.sample_id_list = self.sample_id_list[:configs.num_samples]

        print('Load {} samples from {}'.format(mode, self.dataset_dir))
        print('Done: total {} samples {}'.format(mode, len(self.sample_id_list)))

    def __getitem__(self, index):
        sample_id = int(self.sample_id_list[index])

        if self.mode in ['train', 'val']:
            lidarData = self.get_lidar(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)

            labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)

            if not noObjectLabels:
                labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                                   calib.P)  # convert rect cam to velo cord

            if self.data_aug and self.mode == 'train':
                lidarData, labels[:, 1:] = transformation.complex_yolo_pc_augmentation(lidarData, labels[:, 1:], True)

            b = kitti_bev_utils.removePoints(lidarData, self.configs.boundary)
            rgb_map = kitti_bev_utils.makeBVFeature(b, self.configs.DISCRETIZATION, self.configs.boundary)
            target = kitti_bev_utils.build_yolo_target(labels)
            img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))

            ntargets = 0
            for i, t in enumerate(target):
                if t.sum(0):
                    ntargets += 1
            # (box_idx, class, y, x, w, l, sin(yaw), cos(yaw))
            targets = torch.zeros((ntargets, 8))
            for i, t in enumerate(target):
                if t.sum(0):
                    targets[i, 1:] = torch.from_numpy(t)

            img = torch.from_numpy(rgb_map).float()

            if self.data_aug:
                if np.random.random() < 0.5:
                    img, targets = self.horizontal_flip(img, targets)

            return img_file, img, targets

        else:
            lidarData = self.get_lidar(sample_id)
            b = kitti_bev_utils.removePoints(lidarData, self.configs.boundary)
            rgb_map = kitti_bev_utils.makeBVFeature(b, self.configs.DISCRETIZATION, self.configs.boundary)
            img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))
            return img_file, rgb_map

    def __len__(self):
        return len(self.sample_id_list)

    def preprocess_training_data(self):
        """
        Discard samples which don't have current training class objects, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        for sample_id in self.image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)
            if not noObjectLabels:
                labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                                   calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in self.configs.CLASS_NAME_TO_ID.values():
                    if self.check_point_cloud_range(labels[i, 1:4]):
                        valid_list.append(labels[i, 0])

            if len(valid_list) > 0:
                self.sample_id_list.append(sample_id)

    def check_point_cloud_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [self.configs.boundary["minX"], self.configs.boundary["maxX"]]
        y_range = [self.configs.boundary["minY"], self.configs.boundary["maxY"]]
        z_range = [self.configs.boundary["minZ"], self.configs.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def horizontal_flip(self, images, targets):
        images = torch.flip(images, [-1])
        targets[:, 2] = 1 - targets[:, 2]  # horizontal flip
        targets[:, 6] = - targets[:, 6]  # yaw angle flip

        return images, targets

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        assert os.path.isfile(img_file)
        return cv2.imread(img_file)  # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        assert os.path.isfile(calib_file)
        return kitti_data_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        assert os.path.isfile(label_file)
        return kitti_data_utils.read_label(label_file)

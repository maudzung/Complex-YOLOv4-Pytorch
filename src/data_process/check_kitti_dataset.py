from __future__ import division

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

import data_process.kitti_bev_utils as bev_utils
from data_process.kitti_dataset import KittiDataset
from config.config import parse_configs

if __name__ == "__main__":
    configs = parse_configs()
    img_size = configs.BEV_WIDTH

    # Get dataloader
    dataset = KittiDataset(
        configs,
        split='valid',
        mode='TRAIN',
        data_aug=True,
        multiscale=False
    )

    # Load Dataset
    dataloader = DataLoader(
        dataset,
        1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    for batch_i, (_, imgs, targets) in enumerate(dataloader):

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

        if cv2.waitKey(0) & 0xff == 27:
            break

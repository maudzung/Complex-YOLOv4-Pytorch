"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.20
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for iou calculation of rotated boxes (on GPU)

"""

from __future__ import division
import sys
from math import pi

import torch

sys.path.append('../')
from utils.cal_intersection_rotated_boxes import intersection_area
from utils.convex_hull_torch import convex_hull


def PolyArea2D_torch(pts):
    lines = torch.cat([pts, torch.roll(pts, -1, dims=0)], dim=1)
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
    return area


def get_corners_vectorize(x, y, w, l, yaw):
    """bev image coordinates format - vectorization

    :param box2: [num_boxes, 6]
    :return: num_boxes x (x,y) of 4 conners
    """
    device = x.device
    bbox2 = torch.zeros((x.size(0), 4, 2), device=device, dtype=torch.float)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # front left
    bbox2[:, 0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bbox2[:, 0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bbox2[:, 1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bbox2[:, 1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bbox2[:, 2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bbox2[:, 2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bbox2[:, 3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bbox2[:, 3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bbox2


def iou_rotated_boxes_targets_vs_anchors(targets, anchors, fix_xy=100):
    num_targets_boxes = len(targets)
    targets_areas = targets.t()[0] * targets.t()[1]

    num_anchors = len(anchors)
    anchors_areas = anchors.t()[0] * anchors.t()[1]

    ious = torch.zeros(size=(num_anchors, num_targets_boxes), dtype=torch.float)
    for a_idx in range(num_anchors):
        for tg_idx in range(num_targets_boxes):
            t_w, t_h, t_im, t_re = targets[tg_idx]
            box1 = [fix_xy, fix_xy, t_w, t_h, torch.atan2(t_im, t_re)]  # scale up w and h

            a_w, a_h, a_im, a_re = anchors[a_idx]
            box2 = [fix_xy, fix_xy, a_w, a_h, torch.atan2(a_im, a_re)]

            intersection = intersection_area(box1, box2)
            iou = intersection / (anchors_areas[a_idx] + targets_areas[tg_idx] - intersection + 1e-12)
            ious[a_idx, tg_idx] = iou

    return ious


def iou_pred_vs_target_boxes(pred_boxes, target_boxes, GIoU=False, DIoU=False, CIoU=False):
    assert pred_boxes.size() == target_boxes.size(), "Unmatch size of pred_boxes and target_boxes"
    device = pred_boxes.device
    n_boxes = pred_boxes.size(0)

    t_x, t_y, t_w, t_l, t_im, t_re = target_boxes.t()
    t_yaw = torch.atan2(t_im, t_re)
    target_conners = get_corners_vectorize(t_x, t_y, t_w, t_l, t_yaw)
    target_areas = t_w * t_l

    pred_x, pred_y, pred_w, pred_l, pred_im, pred_re = pred_boxes.t()
    pred_yaw = torch.atan2(pred_im, pred_re)
    pred_conners = get_corners_vectorize(pred_x, pred_y, pred_w, pred_l, pred_yaw)
    pred_areas = pred_w * pred_l

    ious = []
    giou_loss = torch.tensor([0.], device=device, dtype=torch.float)
    # Thinking to apply vectorization this step
    for box_idx in range(n_boxes):
        pred_cons, t_cons = pred_conners[box_idx], target_conners[box_idx]
        pred_area, t_area = pred_areas[box_idx], target_areas[box_idx]
        t_x, t_y, t_w, t_l, t_im, t_re = target_boxes[box_idx]
        pred_x, pred_y, pred_w, pred_l, pred_im, pred_re = pred_boxes[box_idx]
        box1 = [t_x, t_y, t_w, t_l, torch.atan2(t_im, t_re)]
        box2 = [pred_x, pred_y, pred_w, pred_l, torch.atan2(pred_im, pred_re)]
        intersection = intersection_area(box1, box2)
        union = pred_area + t_area - intersection
        iou = intersection / (union + 1e-16)

        if GIoU:
            convex_conners = torch.cat((pred_cons, t_cons), dim=0)
            convex_conners = convex_hull(convex_conners)
            convex_area = PolyArea2D_torch(convex_conners)
            giou_loss += 1. - (iou - (convex_area - union) / (convex_area + 1e-16))
        else:
            giou_loss += 1. - iou

        if DIoU or CIoU:
            raise NotImplementedError

        ious.append(iou)

    return torch.tensor(ious, device=device, dtype=torch.float), giou_loss


if __name__ == "__main__":
    import cv2
    import numpy as np


    def get_corners_torch(x, y, w, l, yaw):
        device = x.device
        bev_corners = torch.zeros((4, 2), dtype=torch.float, device=device)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        # front left
        bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
        bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

        # rear left
        bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
        bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

        # rear right
        bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
        bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

        # front right
        bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
        bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

        return bev_corners


    # Show convex in an image

    img_size = 300
    img = np.zeros((img_size, img_size, 3))
    img = cv2.resize(img, (img_size, img_size))

    box1 = torch.tensor([100, 100, 40, 20, pi / 2], dtype=torch.float).cuda(1)
    box2 = torch.tensor([100, 100, 40, 20, 0], dtype=torch.float).cuda(1)

    intersection = intersection_area(box1, box2)
    print('intersection: {}'.format(intersection))

    box1_conners = get_corners_torch(box1[0], box1[1], box1[2], box1[3], box1[4])
    box2_conners = get_corners_torch(box2[0], box2[1], box2[2], box2[3], box2[4])
    print('PolyArea2D of box1: {}, box2: {}'.format(PolyArea2D_torch(box1_conners), PolyArea2D_torch(box2_conners)))

    merger_conners_torch = convex_hull(torch.cat((box1_conners, box2_conners), dim=0))
    print('merger_conners_torch area: {}'.format(PolyArea2D_torch(merger_conners_torch)))

    img = cv2.polylines(img, [box1_conners.cpu().numpy().astype(np.int)], True, (255, 0, 0), 2)
    img = cv2.polylines(img, [box2_conners.cpu().numpy().astype(np.int)], True, (0, 255, 0), 2)
    img = cv2.polylines(img, [merger_conners_torch.cpu().numpy().astype(np.int)], True, (0, 0, 255), 2)

    while True:
        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xff == 27:
            break

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

import torch
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull


def cvt_box_2_polygon(box):
    """
    :param array: an array of shape [num_conners, 2]
    :return: a shapely.geometry.Polygon object
    """
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)


def get_corners_vectorize(x, y, w, l, yaw):
    """bev image coordinates format - vectorization

    :param x, y, w, l, yaw: [num_boxes,]
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


def get_polygons_areas_fix_xy(boxes, fix_xy=100.):
    """
    Args:
        box: (num_boxes, 4) --> w, l, im, re
    """
    device = boxes.device
    n_boxes = boxes.size(0)
    x = torch.full(size=(n_boxes,), fill_value=fix_xy, device=device, dtype=torch.float)
    y = torch.full(size=(n_boxes,), fill_value=fix_xy, device=device, dtype=torch.float)
    w, l, im, re = boxes.t()
    yaw = torch.atan2(im, re)
    boxes_conners = get_corners_vectorize(x, y, w, l, yaw)
    boxes_polygons = [cvt_box_2_polygon(box_) for box_ in boxes_conners]
    boxes_areas = w * l

    return boxes_polygons, boxes_areas


def iou_rotated_boxes_targets_vs_anchors(anchors_polygons, anchors_areas, targets_polygons, targets_areas):
    device = anchors_areas.device
    num_anchors = len(anchors_areas)
    num_targets_boxes = len(targets_areas)

    ious = torch.zeros(size=(num_anchors, num_targets_boxes), device=device, dtype=torch.float)

    for a_idx in range(num_anchors):
        for tg_idx in range(num_targets_boxes):
            intersection = anchors_polygons[a_idx].intersection(targets_polygons[tg_idx]).area
            iou = intersection / (anchors_areas[a_idx] + targets_areas[tg_idx] - intersection + 1e-16)
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
        intersection = cvt_box_2_polygon(pred_cons).intersection(cvt_box_2_polygon(t_cons)).area
        union = pred_area + t_area - intersection
        iou = intersection / (union + 1e-16)

        if GIoU:
            convex_conners = torch.cat((pred_cons, t_cons), dim=0)
            hull = ConvexHull(convex_conners.clone().detach().cpu().numpy())  # done on cpu, just need indices output
            convex_conners = convex_conners[hull.vertices]
            convex_polygon = cvt_box_2_polygon(convex_conners)
            convex_area = convex_polygon.area
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

    box1 = torch.tensor([100, 100, 40, 20, np.pi / 2], dtype=torch.float).cuda()
    box2 = torch.tensor([100, 100, 40, 20, 0], dtype=torch.float).cuda()

    box1_conners = get_corners_torch(box1[0], box1[1], box1[2], box1[3], box1[4])
    box1_polygon = cvt_box_2_polygon(box1_conners)
    box1_area = box1_polygon.area

    box2_conners = get_corners_torch(box2[0], box2[1], box2[2], box2[3], box2[4])
    box2_polygon = cvt_box_2_polygon(box2_conners)
    box2_area = box2_polygon.area

    intersection = box2_polygon.intersection(box1_polygon).area
    union = box1_area + box2_area - intersection
    iou = intersection / (union + 1e-16)

    convex_conners = torch.cat((box1_conners, box2_conners), dim=0)
    hull = ConvexHull(convex_conners.clone().detach().cpu().numpy())  # done on cpu, just need indices output
    convex_conners = convex_conners[hull.vertices]
    convex_polygon = cvt_box_2_polygon(convex_conners)
    convex_area = convex_polygon.area
    giou_loss = 1. - (iou - (convex_area - union) / (convex_area + 1e-16))

    print(
        'box1_area: {:.2f}, box2_area: {:.2f}, intersection: {:.2f}, iou: {:.4f}, convex_area: {:.4f}, giou_loss: {}'.format(
            box1_area, box2_area, intersection, iou, convex_area, giou_loss))

    img = cv2.polylines(img, [box1_conners.cpu().numpy().astype(np.int)], True, (255, 0, 0), 2)
    img = cv2.polylines(img, [box2_conners.cpu().numpy().astype(np.int)], True, (0, 255, 0), 2)
    img = cv2.polylines(img, [convex_conners.cpu().numpy().astype(np.int)], True, (0, 0, 255), 2)

    while True:
        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xff == 27:
            break

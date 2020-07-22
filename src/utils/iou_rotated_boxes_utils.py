from __future__ import division
import sys

import torch
import numpy as np
from shapely.geometry import Polygon

sys.path.append('../')

import data_process.kitti_bev_utils as bev_utils
from utils.torch_utils import to_cpu


def cvt_box_2_polygon(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]) for box in boxes_array]

    return np.array(polygons)


def compute_iou_polygons(polygon_1, polygons):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [polygon_1.intersection(poly_).area / (polygon_1.union(poly_).area + 1e-12) for poly_ in polygons]

    return np.array(iou, dtype=np.float32)


def iou_rotated_boxes_vs_anchors(anchors_polygons, anchors_areas, targets_polygons, targets_areas):
    num_anchors = len(anchors_areas)
    num_targets_boxes = len(targets_areas)
    ious = torch.zeros(size=(num_anchors, num_targets_boxes), dtype=torch.float)
    for ac_idx in range(num_anchors):
        for tg_idx in range(num_targets_boxes):
            intersection = anchors_polygons[ac_idx].intersection(targets_polygons[tg_idx]).area
            iou = intersection / (anchors_areas[ac_idx] + targets_areas[tg_idx] - intersection + 1e-12)
            ious[ac_idx, tg_idx] = iou

    return ious


def get_polygons_fix_xy(boxes, fix_xy=100):
    """
    Args:
        box: (num_boxes, 4) --> w, l, im, re
    """

    n_boxes = boxes.shape[0]
    x = np.full(shape=(n_boxes,), fill_value=fix_xy, dtype=np.float32)
    y = np.full(shape=(n_boxes,), fill_value=fix_xy, dtype=np.float32)
    w, l, im, re = boxes.transpose(1, 0)
    yaw = np.arctan2(im, re)
    ret_conners = bev_utils.get_corners_vectorize(x, y, w, l, yaw)
    ret_polygons = cvt_box_2_polygon(ret_conners)

    return ret_polygons


def iou_pred_vs_target_boxes(pred_boxes, target_boxes, GIoU=False, DIoU=False, CIoU=False):
    assert pred_boxes.size() == target_boxes.size(), "Unmatch size of pred_boxes and target_boxes"
    device = pred_boxes.device
    pred_boxes_cpu = to_cpu(pred_boxes).numpy()
    target_boxes_cpu = to_cpu(target_boxes).numpy()

    ious = []
    # Thinking to apply vectorization this step
    for pred_box, target_box in zip(pred_boxes_cpu, target_boxes_cpu):
        iou = iou_rotated_11_boxes(pred_box, target_box)
        if GIoU or DIoU or CIoU:
            raise NotImplementedError

        ious.append(iou)

    return torch.tensor(ious, device=device, dtype=torch.float)


def iou_rotated_11_boxes(box1, box2):
    x, y, w, l, im, re = box1
    yaw = np.arctan2(im, re)
    bbox1 = bev_utils.get_corners(x, y, w, l, yaw)
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    box1_polygon = Polygon([(bbox1[i, 0], bbox1[i, 1]) for i in range(len(bbox1))]).buffer(0)

    x, y, w, l, im, re = box2
    yaw = np.arctan2(im, re)
    bbox2 = bev_utils.get_corners(x, y, w, l, yaw)
    box2_polygon = Polygon([(bbox2[i, 0], bbox2[i, 1]) for i in range(len(bbox2))]).buffer(0)  # to fix a line polygon
    iou = box1_polygon.intersection(box2_polygon).area / (box1_polygon.union(box2_polygon).area + 1e-12)

    return iou

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


def iou_rotated_boxes_vs_anchor(anchor, wh, imre):
    """

    Args:
        anchor: (num anchors, 4)
        wh: (num_boxes, 2)
        imre: (num_boxes, 2)

    Returns:

    """
    # last dimension has 6: x, y, w, l, im, re
    anchor_box = np.full(shape=(anchor.shape[0], 6), fill_value=100, dtype=np.float32)
    anchor_box[:, 2:6] = to_cpu(anchor).numpy()

    target_boxes = np.full(shape=(wh.shape[0], 6), fill_value=100, dtype=np.float32)
    target_boxes[:, 2:4] = to_cpu(wh).numpy()
    target_boxes[:, 4:6] = to_cpu(imre).numpy()

    ious = iou_rotated_boxes(anchor_box[0], target_boxes)

    return torch.from_numpy(ious)


def iou_rotated_boxes(box1, box2):
    """ calculate IoU of polygons with vectorization

    :param box1: (6,)
    :param box2: (num, 6)
    :return:
    """
    x, y, w, l, im, re = box1
    yaw = np.arctan2(im, re)
    bbox1 = np.array(bev_utils.get_corners(x, y, w, l, yaw)).reshape(-1, 4, 2)
    box1_polygon = cvt_box_2_polygon(bbox1)

    x, y, w, l, im, re = box2.transpose(1, 0)
    yaw = np.arctan2(im, re)
    bbox2 = bev_utils.get_corners_vectorize(x, y, w, l, yaw)
    box2_polygons = cvt_box_2_polygon(bbox2)

    return compute_iou_polygons(box1_polygon[0], box2_polygons)


def iou_pred_vs_target_boxes(pred_boxes, target_boxes, nG, GIoU=False, DIoU=False, CIoU=False):
    assert pred_boxes.size() == target_boxes.size(), "Unmatch size of pred_boxes and target_boxes"
    device = pred_boxes.device
    pred_boxes_cp = np.copy(to_cpu(pred_boxes).numpy())
    target_boxes_cp = np.copy(to_cpu(target_boxes).numpy())

    target_boxes_cp[:, :4] = target_boxes_cp[:, :4] * nG

    ious = []
    # Thinking to apply vectorization this step
    for pred_box, target_box in zip(pred_boxes_cp, target_boxes_cp):
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

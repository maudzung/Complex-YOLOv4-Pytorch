from __future__ import division
import sys
import tqdm

import torch
import numpy as np
from shapely.geometry import Polygon

sys.path.append('../')

import data_process.kitti_bev_utils as bev_utils


def cvt_box_2_polygon(boxes_array):
    """
    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """
    # use .buffer(0) to fix a line polygon
    # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0) for box in boxes_array]

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


def compute_iou_nms(idx_self, idx_other, polygons, areas):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    ious = []
    box1 = polygons[idx_self]
    for idx in idx_other:
        box2 = polygons[idx]
        intersection = box1.intersection(box2).area
        iou = intersection / (areas[idx] + areas[idx_self] - intersection + 1e-12)
        ious.append(iou)

    return np.array(ious, dtype=np.float32)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :6]
        pred_scores = output[:, 6]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                # iou, box_index = rotated_bbox_iou(pred_box.unsqueeze(0), target_boxes, 1.0, False).squeeze().max(0)
                ious = rotated_bbox_iou_polygon_cpu(pred_box, target_boxes)
                iou, box_index = torch.from_numpy(ious).max(0)

                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def rotated_bbox_iou_polygon_cpu(box1, box2):
    """
    :param box1: Numpy array
    :param box2: Numpy array
    :return:
    """

    x, y, w, l, im, re = box1
    angle = np.arctan2(im, re)
    bbox1 = np.array(bev_utils.get_corners(x, y, w, l, angle)).reshape(-1, 4, 2)
    bbox1 = cvt_box_2_polygon(bbox1)

    bbox2 = []
    for i in range(box2.shape[0]):
        x, y, w, l, im, re = box2[i, :]
        angle = np.arctan2(im, re)
        bev_corners = bev_utils.get_corners(x, y, w, l, angle)
        bbox2.append(bev_corners)
    bbox2 = cvt_box_2_polygon(np.array(bbox2))

    return compute_iou_polygons(bbox1[0], bbox2)


def compute_polygons(boxes):
    """

    :param boxes: [num, 6]
    :return:
    """
    polygons = []
    for (x, y, w, l, im, re) in boxes:
        angle = np.arctan2(im, re)
        bev_corners = bev_utils.get_corners(x, y, w, l, angle)
        polygons.append(bev_corners)
    polygons = cvt_box_2_polygon(np.array(polygons))

    return polygons


def nms_cpu(boxes, confs, nms_thresh=0.5):
    """

    :param boxes: [num, 6]
    :param confs: [num, num_classes]
    :param nms_thresh:
    :param min_mode:
    :return:
    """
    # order of reduce confidence (high --> low)
    order = confs.argsort()[::-1]

    polygons = compute_polygons(boxes)  # 4 vertices of the box

    areas = [polygon.area for polygon in polygons]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]
        keep.append(idx_self)
        over = compute_iou_nms(idx_self, idx_other, polygons, areas)
        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def post_processing(outputs, conf_thresh=0.95, nms_thresh=0.4):
    """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape:
            (x, y, w, l, im, re, object_conf, class_score, class_pred)
    """
    if type(outputs).__name__ != 'ndarray':
        outputs = outputs.numpy()
    # outputs shape: (batch_size, 22743, 10)
    batch_size = outputs.shape[0]
    # box_array: [batch, num, 6]
    box_array = outputs[:, :, :6]

    # confs: [batch, num, num_classes]
    confs = outputs[:, :, 6:7] * outputs[:, :, 7:]

    # [batch, num, num_classes] --> [batch, num]
    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = [None for _ in range(batch_size)]

    for i in range(batch_size):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        keep = nms_cpu(l_box_array, l_max_conf, nms_thresh=nms_thresh)

        bboxes = []
        if (keep.size > 0):
            l_box_array = l_box_array[keep, :]
            l_max_conf = l_max_conf[keep]
            l_max_id = l_max_id[keep]

            for j in range(l_box_array.shape[0]):
                bboxes.append(
                    [l_box_array[j, 0], l_box_array[j, 1], l_box_array[j, 2], l_box_array[j, 3], l_box_array[j, 4],
                     l_box_array[j, 5], l_max_conf[j], l_max_id[j]])
        if len(bboxes) > 0:
            bboxes_batch[i] = np.array(bboxes)

    return bboxes_batch


if __name__ == '__main__':
    import time

    prediction = torch.randn((4, 22743, 10))
    print('prediction size: {}'.format(prediction.size()))
    output = post_processing(prediction, conf_thresh=0.99999, nms_thresh=0.9999)

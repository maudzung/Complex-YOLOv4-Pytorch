"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the yolo layer

# Refer: https://github.com/Tianxiaomo/pytorch-YOLOv4
# Refer: https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')

from utils.torch_utils import to_cpu
from utils.iou_rotated_boxes_utils import iou_pred_vs_target_boxes, iou_rotated_boxes_targets_vs_anchors, \
    get_polygons_areas_fix_xy


class YoloLayer(nn.Module):
    """Yolo layer"""

    def __init__(self, num_classes, anchors, stride, scale_x_y, ignore_thresh):
        super(YoloLayer, self).__init__()
        # Update the attributions when parsing the cfg during create the darknet
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.stride = stride
        self.scale_x_y = scale_x_y
        self.ignore_thresh = ignore_thresh

        self.noobj_scale = 100
        self.obj_scale = 1
        self.lgiou_scale = 3.54
        self.leular_scale = 3.54
        self.lobj_scale = 64.3
        self.lcls_scale = 37.4

        self.seen = 0
        # Initialize dummy variables
        self.grid_size = 0
        self.img_size = 0
        self.metrics = {}

    def compute_grid_offsets(self, grid_size):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_size / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g, device=self.device, dtype=torch.float).repeat(g, 1).view([1, 1, g, g])
        self.grid_y = torch.arange(g, device=self.device, dtype=torch.float).repeat(g, 1).t().view([1, 1, g, g])
        self.scaled_anchors = torch.tensor(
            [(a_w / self.stride, a_h / self.stride, im, re) for a_w, a_h, im, re in self.anchors], device=self.device,
            dtype=torch.float)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Pre compute polygons and areas of anchors
        self.scaled_anchors_polygons, self.scaled_anchors_areas = get_polygons_areas_fix_xy(self.scaled_anchors)

    def build_targets(self, pred_boxes, pred_cls, target, anchors):
        """ Built yolo targets to compute loss
        :param out_boxes: [num_samples or batch, num_anchors, grid_size, grid_size, 6]
        :param pred_cls: [num_samples or batch, num_anchors, grid_size, grid_size, num_classes]
        :param target: [num_boxes, 8]
        :param anchors: [num_anchors, 4]
        :return:
        """
        nB, nA, nG, _, nC = pred_cls.size()
        n_target_boxes = target.size(0)

        # Create output tensors on "device"
        obj_mask = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.uint8)
        noobj_mask = torch.full(size=(nB, nA, nG, nG), fill_value=1, device=self.device, dtype=torch.uint8)
        class_mask = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        iou_scores = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tx = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        ty = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tw = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        th = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tim = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tre = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=self.device, dtype=torch.float)
        tcls = torch.full(size=(nB, nA, nG, nG, nC), fill_value=0, device=self.device, dtype=torch.float)
        tconf = obj_mask.float()
        giou_loss = torch.tensor([0.], device=self.device, dtype=torch.float)

        if n_target_boxes > 0:  # Make sure that there is at least 1 box
            b, target_labels = target[:, :2].long().t()
            target_boxes = torch.cat((target[:, 2:6] * nG, target[:, 6:8]), dim=-1)  # scale up x, y, w, h

            gxy = target_boxes[:, :2]
            gwh = target_boxes[:, 2:4]
            gimre = target_boxes[:, 4:6]

            targets_polygons, targets_areas = get_polygons_areas_fix_xy(target_boxes[:, 2:6])
            # Get anchors with best iou
            ious = iou_rotated_boxes_targets_vs_anchors(self.scaled_anchors_polygons, self.scaled_anchors_areas,
                                                        targets_polygons, targets_areas)
            best_ious, best_n = ious.max(0)

            gx, gy = gxy.t()
            gw, gh = gwh.t()
            gim, gre = gimre.t()
            gi, gj = gxy.long().t()
            # Set masks
            obj_mask[b, best_n, gj, gi] = 1
            noobj_mask[b, best_n, gj, gi] = 0

            # Set noobj mask to zero where iou exceeds ignore threshold
            for i, anchor_ious in enumerate(ious.t()):
                noobj_mask[b[i], anchor_ious > self.ignore_thresh, gj[i], gi[i]] = 0

            # Coordinates
            tx[b, best_n, gj, gi] = gx - gx.floor()
            ty[b, best_n, gj, gi] = gy - gy.floor()
            # Width and height
            tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
            th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
            # Im and real part
            tim[b, best_n, gj, gi] = gim
            tre[b, best_n, gj, gi] = gre

            # One-hot encoding of label
            tcls[b, best_n, gj, gi, target_labels] = 1
            class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
            ious, giou_loss = iou_pred_vs_target_boxes(pred_boxes[b, best_n, gj, gi], target_boxes,
                                                       GIoU=self.use_giou_loss)
            iou_scores[b, best_n, gj, gi] = ious
            if self.reduction == 'mean':
                giou_loss /= n_target_boxes
            tconf = obj_mask.float()

        return iou_scores, giou_loss, class_mask, obj_mask.type(torch.bool), noobj_mask.type(torch.bool), \
               tx, ty, tw, th, tim, tre, tcls, tconf

    def forward(self, x, targets=None, img_size=608, use_giou_loss=False):
        """
        :param x: [num_samples or batch, num_anchors * (6 + 1 + num_classes), grid_size, grid_size]
        :param targets: [num boxes, 8] (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
        :param img_size: default 608
        :return:
        """
        self.img_size = img_size
        self.use_giou_loss = use_giou_loss
        self.device = x.device
        num_samples, _, _, grid_size = x.size()

        prediction = x.view(num_samples, self.num_anchors, self.num_classes + 7, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        # prediction size: [num_samples, num_anchors, grid_size, grid_size, num_classes + 7]

        # Get outputs
        pred_x = torch.sigmoid(prediction[..., 0])
        pred_y = torch.sigmoid(prediction[..., 1])
        pred_w = prediction[..., 2]  # Width
        pred_h = prediction[..., 3]  # Height
        pred_im = prediction[..., 4]  # angle imaginary part
        pred_re = prediction[..., 5]  # angle real part
        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size)

        # Add offset and scale with anchors
        # pred_boxes size: [num_samples, num_anchors, grid_size, grid_size, 6]
        pred_boxes = torch.empty(prediction[..., :6].shape, device=self.device, dtype=torch.float)
        pred_boxes[..., 0] = pred_x + self.grid_x
        pred_boxes[..., 1] = pred_y + self.grid_y
        pred_boxes[..., 2] = torch.exp(pred_w).clamp(max=1E3) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(pred_h).clamp(max=1E3) * self.anchor_h
        pred_boxes[..., 4] = pred_im
        pred_boxes[..., 5] = pred_re

        output = torch.cat((
            pred_boxes[..., :4].view(num_samples, -1, 4) * self.stride,
            pred_boxes[..., 4:6].view(num_samples, -1, 2),
            pred_conf.view(num_samples, -1, 1),
            pred_cls.view(num_samples, -1, self.num_classes),
        ), dim=-1)
        # output size: [num_samples, num boxes, 7 + num_classes]

        if targets is None:
            return output, 0
        else:
            self.reduction = 'mean'
            iou_scores, giou_loss, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes, pred_cls=pred_cls, target=targets, anchors=self.scaled_anchors)

            loss_x = F.mse_loss(pred_x[obj_mask], tx[obj_mask], reduction=self.reduction)
            loss_y = F.mse_loss(pred_y[obj_mask], ty[obj_mask], reduction=self.reduction)
            loss_w = F.mse_loss(pred_w[obj_mask], tw[obj_mask], reduction=self.reduction)
            loss_h = F.mse_loss(pred_h[obj_mask], th[obj_mask], reduction=self.reduction)
            loss_im = F.mse_loss(pred_im[obj_mask], tim[obj_mask], reduction=self.reduction)
            loss_re = F.mse_loss(pred_re[obj_mask], tre[obj_mask], reduction=self.reduction)
            loss_im_re = (1. - torch.sqrt(pred_im[obj_mask] ** 2 + pred_re[obj_mask] ** 2)) ** 2  # as tim^2 + tre^2 = 1
            loss_im_re_red = loss_im_re.sum() if self.reduction == 'sum' else loss_im_re.mean()
            loss_eular = loss_im + loss_re + loss_im_re_red

            loss_conf_obj = F.binary_cross_entropy(pred_conf[obj_mask], tconf[obj_mask], reduction=self.reduction)
            loss_conf_noobj = F.binary_cross_entropy(pred_conf[noobj_mask], tconf[noobj_mask], reduction=self.reduction)
            loss_cls = F.binary_cross_entropy(pred_cls[obj_mask], tcls[obj_mask], reduction=self.reduction)

            if self.use_giou_loss:
                loss_obj = loss_conf_obj + loss_conf_noobj
                total_loss = giou_loss * self.lgiou_scale + loss_eular * self.leular_scale + loss_obj * self.lobj_scale + loss_cls * self.lcls_scale
            else:
                loss_obj = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                total_loss = loss_x + loss_y + loss_w + loss_h + loss_eular + loss_obj + loss_cls

                # Metrics (store loss values using tensorboard)
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "iou_score": to_cpu(iou_scores[obj_mask].mean()).item(),
                'giou_loss': to_cpu(giou_loss).item(),
                'loss_x': to_cpu(loss_x).item(),
                'loss_y': to_cpu(loss_y).item(),
                'loss_w': to_cpu(loss_w).item(),
                'loss_h': to_cpu(loss_h).item(),
                'loss_eular': to_cpu(loss_eular).item(),
                'loss_im': to_cpu(loss_im).item(),
                'loss_re': to_cpu(loss_re).item(),
                "loss_obj": to_cpu(loss_obj).item(),
                "loss_cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item()
            }

            return output, total_loss

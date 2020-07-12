"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/Tianxiaomo/pytorch-YOLOv4
"""

import sys

import torch
import torch.nn as nn

sys.path.append('../')

from utils.torch_utils import to_cpu
from utils.evaluation_utils import rotated_box_11_iou_polygon, rotated_box_wh_iou_polygon


class YoloLayer(nn.Module):
    """Yolo layer"""

    def __init__(self, num_classes, anchors, stride, scale_x_y, ignore_thresh):
        super(YoloLayer, self).__init__()
        # Update the attributions when parsing the cfg during create the darknet
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.coord_scale = 1
        self.noobj_scale = 1
        self.obj_scale = 5
        self.class_scale = 1
        self.ignore_thresh = ignore_thresh
        self.stride = stride
        self.seen = 0
        self.scale_x_y = scale_x_y
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        # Initialize dummy variables
        self.grid_size = 0
        self.img_size = 0
        self.metrics = {}

    def compute_grid_offsets(self, grid_size, device):
        self.grid_size = grid_size
        g = self.grid_size
        self.stride = self.img_size / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g, device=device, dtype=torch.float).repeat(g, 1).view([1, 1, g, g])
        self.grid_y = torch.arange(g, device=device, dtype=torch.float).repeat(g, 1).t().view([1, 1, g, g])
        self.scaled_anchors = torch.tensor(
            [(a_w / self.stride, a_h / self.stride, im, re) for a_w, a_h, im, re in self.anchors], device=device,
            dtype=torch.float)
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def build_targets(self, pred_boxes, pred_cls, target, anchors, device):
        """ Built yolo targets to compute loss

        :param pred_boxes: [num_samples or batch, num_anchors, grid_size, grid_size, 6]
        :param pred_cls: [num_samples or batch, num_anchors, grid_size, grid_size, 3]
        :param target: [num_boxes, 8]
        :param anchors: [num_anchors, 4]
        :return:
        """

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)

        # Create output tensors on "device"
        obj_mask = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.uint8)
        noobj_mask = torch.full(size=(nB, nA, nG, nG), fill_value=1, device=device, dtype=torch.uint8)
        class_mask = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.float)
        iou_scores = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.float)
        tx = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.float)
        ty = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.float)
        tw = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.float)
        th = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.float)
        tim = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.float)
        tre = torch.full(size=(nB, nA, nG, nG), fill_value=0, device=device, dtype=torch.float)
        tcls = torch.full(size=(nB, nA, nG, nG, nC), fill_value=0, device=device, dtype=torch.float)

        # Convert to position relative to box
        target_boxes = target[:, 2:8]

        gxy = target_boxes[:, :2] * nG
        gwh = target_boxes[:, 2:4] * nG
        gimre = target_boxes[:, 4:]

        # Get anchors with best iou
        ious = torch.stack([rotated_box_wh_iou_polygon(anchor, gwh, gimre) for anchor in anchors])

        best_ious, best_n = ious.max(0)
        b, target_labels = target[:, :2].long().t()

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
        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

        rotated_iou_scores = rotated_box_11_iou_polygon(pred_boxes[b, best_n, gj, gi], target_boxes, nG)
        iou_scores[b, best_n, gj, gi] = rotated_iou_scores.cuda()

        tconf = obj_mask.float()
        return iou_scores, class_mask, obj_mask.type(torch.bool), noobj_mask.type(
            torch.bool), tx, ty, tw, th, tim, tre, tcls, tconf

    def forward(self, x, targets=None, img_size=608, device=None):
        """
        :param x: [num_samples or batch, num_anchors * (6 + 1 + num_classes), grid_size, grid_size]
        :param targets: [num boxes, 8] (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
        :param img_size: default 608
        :return:
        """
        self.img_size = img_size
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = x.view(num_samples, self.num_anchors, self.num_classes + 7, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        # prediction size: [num_samples, num_anchors, grid_size, grid_size, num_classes + 7]

        # Get outputs
        x = torch.sigmoid(prediction[..., 0]) * self.scale_x_y - 0.5 * (self.scale_x_y - 1)  # Center x
        y = torch.sigmoid(prediction[..., 1]) * self.scale_x_y - 0.5 * (self.scale_x_y - 1)  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        im = prediction[..., 4]  # angle imaginary part
        re = prediction[..., 5]  # angle real part
        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, device)

        # Add offset and scale with anchors
        # pred_boxes size: [num_samples, num_anchors, grid_size, grid_size, 6]
        pred_boxes = torch.empty(prediction[..., :6].shape, device=device, dtype=torch.float)
        pred_boxes[..., 0] = x.detach() + self.grid_x
        pred_boxes[..., 1] = y.detach() + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.detach()) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.detach()) * self.anchor_h
        pred_boxes[..., 4] = im
        pred_boxes[..., 5] = re

        output = torch.cat((
            pred_boxes[..., :4].view(num_samples, -1, 4) * self.stride,
            pred_boxes[..., 4:].view(num_samples, -1, 2),
            pred_conf.view(num_samples, -1, 1),
            pred_cls.view(num_samples, -1, self.num_classes),
        ), dim=-1)
        # output size: [num_samples, num boxes, 7 + num_classes]

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tim, tre, tcls, tconf = self.build_targets(
                pred_boxes=pred_boxes, pred_cls=pred_cls, target=targets, anchors=self.scaled_anchors, device=device)

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            loss_im = self.mse_loss(im[obj_mask], tim[obj_mask])
            loss_re = self.mse_loss(re[obj_mask], tre[obj_mask])
            loss_eular = loss_im + loss_re
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_eular + loss_conf + loss_cls

            # Metrics
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
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "im": to_cpu(loss_im).item(),
                "re": to_cpu(loss_re).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }

            return output, total_loss

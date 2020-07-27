"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
"""

import math
import sys

import cv2
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf


def removePoints(PointCloud, BoundaryCond):
    # Boundary condition
    minX = BoundaryCond['minX']
    maxX = BoundaryCond['maxX']
    minY = BoundaryCond['minY']
    maxY = BoundaryCond['maxY']
    minZ = BoundaryCond['minZ']
    maxZ = BoundaryCond['maxZ']

    # Remove the point out of range x,y,z
    mask = np.where((PointCloud[:, 0] >= minX) & (PointCloud[:, 0] <= maxX) & (PointCloud[:, 1] >= minY) & (
            PointCloud[:, 1] <= maxY) & (PointCloud[:, 2] >= minZ) & (PointCloud[:, 2] <= maxZ))
    PointCloud = PointCloud[mask]

    PointCloud[:, 2] = PointCloud[:, 2] - minZ

    return PointCloud


def makeBVFeature(PointCloud_, Discretization, bc):
    Height = cnf.BEV_HEIGHT + 1
    Width = cnf.BEV_WIDTH + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / Discretization))
    PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / Discretization) + Width / 2)

    # sort-3times
    indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
    PointCloud_frac = PointCloud[indices]
    # some important problem is image coordinate is (y,x), not (x,y)
    max_height = float(np.abs(bc['maxZ'] - bc['minZ']))
    heightMap[np.int_(PointCloud_frac[:, 0]), np.int_(PointCloud_frac[:, 1])] = PointCloud_frac[:, 2] / max_height

    # Intensity Map & DensityMap
    intensityMap = np.zeros((Height, Width))
    densityMap = np.zeros((Height, Width))

    _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
    PointCloud_top = PointCloud[indices]

    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = PointCloud_top[:, 3]
    densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

    RGB_Map = np.zeros((3, Height - 1, Width - 1))
    RGB_Map[2, :, :] = densityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # r_map
    RGB_Map[1, :, :] = heightMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # g_map
    RGB_Map[0, :, :] = intensityMap[:cnf.BEV_HEIGHT, :cnf.BEV_WIDTH]  # b_map

    return RGB_Map


def read_labels_for_bevbox(objects):
    bbox_selected = []
    for obj in objects:
        if obj.cls_id != -1:
            bbox = []
            bbox.append(obj.cls_id)
            bbox.extend([obj.t[0], obj.t[1], obj.t[2], obj.h, obj.w, obj.l, obj.ry])
            bbox_selected.append(bbox)

    if len(bbox_selected) == 0:
        labels = np.zeros((1, 8), dtype=np.float32)
        noObjectLabels = True
    else:
        labels = np.array(bbox_selected, dtype=np.float32)
        noObjectLabels = False

    return labels, noObjectLabels


# bev image coordinates format
def get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
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


def build_yolo_target(labels):
    bc = cnf.boundary
    target = []
    for i in range(labels.shape[0]):
        cl, x, y, z, h, w, l, yaw = labels[i]
        # ped and cyc labels are very small, so lets add some factor to height/width
        l = l + 0.3
        w = w + 0.3
        yaw = np.pi * 2 - yaw
        if (bc["minX"] < x < bc["maxX"]) and (bc["minY"] < y < bc["maxY"]):
            y1 = (y - bc["minY"]) / (bc["maxY"] - bc["minY"])  # we should put this in [0,1], so divide max_size  80 m
            x1 = (x - bc["minX"]) / (bc["maxX"] - bc["minX"])  # we should put this in [0,1], so divide max_size  40 m
            w1 = w / (bc["maxY"] - bc["minY"])
            l1 = l / (bc["maxX"] - bc["minX"])
            target.append([cl, y1, x1, w1, l1, math.sin(float(yaw)), math.cos(float(yaw))])

    return np.array(target, dtype=np.float32)


def inverse_yolo_target(targets, bc):
    labels = []
    for t in targets:
        c, y, x, w, l, im, re = t
        z, h = -1.55, 1.5
        if c == 1:
            h = 1.8
        elif c == 2:
            h = 1.4

        y = y * (bc["maxY"] - bc["minY"]) + bc["minY"]
        x = x * (bc["maxX"] - bc["minX"]) + bc["minX"]
        w = w * (bc["maxY"] - bc["minY"])
        l = l * (bc["maxX"] - bc["minX"])
        w -= 0.3
        l -= 0.3
        labels.append([c, x, y, z, h, w, l, - np.arctan2(im, re) - 2 * np.pi])

    return np.array(labels)


# send parameters in bev image coordinates format
def drawRotatedBox(img, x, y, w, l, yaw, color):
    bev_corners = get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)


def draw_box_in_bev(rgb_map, target):
    for j in range(50):
        if (np.sum(target[j, 1:]) == 0): continue
        cls_id = int(target[j][0])
        x = target[j][1] * cnf.BEV_WIDTH
        y = target[j][2] * cnf.BEV_HEIGHT
        w = target[j][3] * cnf.BEV_WIDTH
        l = target[j][4] * cnf.BEV_HEIGHT
        yaw = np.arctan2(target[j][5], target[j][6])
        drawRotatedBox(rgb_map, x, y, w, l, yaw, cnf.colors[cls_id])

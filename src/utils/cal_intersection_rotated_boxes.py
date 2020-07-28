"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.20
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for intersection calculation of rotated boxes (on GPU)

Refer from # https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python?noredirect=1&lq=1
"""

import torch


class Line:
    # ax + by + c = 0
    def __init__(self, p1, p2):
        """

        Args:
            p1: (x, y)
            p2: (x, y)
        """
        self.a = p2[1] - p1[1]
        self.b = p1[0] - p2[0]
        self.c = p2[0] * p1[1] - p2[1] * p1[0]  # cross
        self.device = p1.device

    def cal_values(self, pts):
        return self.a * pts[:, 0] + self.b * pts[:, 1] + self.c

    def find_intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplemented
        w = self.a * other.b - self.b * other.a
        return torch.tensor([(self.b * other.c - self.c * other.b) / w, (self.c * other.a - self.a * other.c) / w],
                            device=self.device)


def intersection_area(rect1, rect2):
    """Calculate the inter

    Args:
        rect1: vertices of the rectangles (4, 2)
        rect2: vertices of the rectangles (4, 2)

    Returns:

    """

    # Use the vertices of the first rectangle as, starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    roll_rect2 = torch.roll(rect2, -1, dims=0)
    for p, q in zip(rect2, roll_rect2):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".
        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = line.cal_values(intersection)
        roll_intersection = torch.roll(intersection, -1, dims=0)
        roll_line_values = torch.roll(line_values, -1, dims=0)
        for s, t, s_value, t_value in zip(intersection, roll_intersection, line_values, roll_line_values):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.find_intersection(Line(s, t))
                new_intersection.append(intersection_point)

        if len(new_intersection) > 0:
            intersection = torch.stack(new_intersection)
        else:
            break

    # Calculate area
    if len(intersection) <= 2:
        return 0.

    return PolyArea2D(intersection)


def PolyArea2D(pts):
    roll_pts = torch.roll(pts, -1, dims=0)
    area = (pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]).sum().abs() * 0.5
    return area


if __name__ == "__main__":
    import cv2
    import numpy as np
    from shapely.geometry import Polygon


    def cvt_box_2_polygon(box):
        """
        :param array: an array of shape [num_conners, 2]
        :return: a shapely.geometry.Polygon object
        """
        # use .buffer(0) to fix a line polygon
        # more infor: https://stackoverflow.com/questions/13062334/polygon-intersection-error-in-shapely-shapely-geos-topologicalerror-the-opera
        return Polygon([(box[i, 0], box[i, 1]) for i in range(len(box))]).buffer(0)


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

    box1 = torch.tensor([100, 100, 40, 10, np.pi / 2], dtype=torch.float).cuda()
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

    print('Shapely- box1_area: {:.2f}, box2_area: {:.2f}, inter: {:.2f}, iou: {:.4f}'.format(box1_area, box2_area,
                                                                                             intersection, iou))

    print('intersection from intersection_area(): {}'.format(intersection_area(box1_conners, box2_conners)))

    img = cv2.polylines(img, [box1_conners.cpu().numpy().astype(np.int)], True, (255, 0, 0), 2)
    img = cv2.polylines(img, [box2_conners.cpu().numpy().astype(np.int)], True, (0, 255, 0), 2)

    while True:
        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xff == 27:
            break

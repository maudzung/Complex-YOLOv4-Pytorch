import sys
import os

import numpy as np
from shapely.geometry import Polygon

sys.path.append('../')

from data_process import transformation, kitti_bev_utils, kitti_data_utils
import config.kitti_config as cnf


class Find_Anchors():
    def __init__(self, dataset_dir, img_size, use_yaw_label=False):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.use_yaw_label = use_yaw_label

        self.lidar_dir = os.path.join(self.dataset_dir, 'training', "velodyne")
        self.image_dir = os.path.join(self.dataset_dir, 'training', "image_2")
        self.calib_dir = os.path.join(self.dataset_dir, 'training', "calib")
        self.label_dir = os.path.join(self.dataset_dir, 'training', "label_2")
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', 'trainval.txt')
        self.image_idx_list = [x.strip() for x in open(split_txt_path).readlines()]

        self.sample_id_list = self.remove_invalid_idx(self.image_idx_list)
        self.boxes_wh = self.load_full_boxes_wh()
        # Take out the total number of boxes
        self.num_boxes = self.boxes_wh.shape[0]
        print("number of sample_id_list: {}, num_boxes: {}".format(len(self.sample_id_list), self.num_boxes))
        # Calculate the polygons and areas
        self.boxes_conners = np.array([kitti_bev_utils.get_corners(0, 0, b[0], b[1], b[2]) for b in self.boxes_wh])
        self.boxes_polygons = [self.cvt_box_2_polygon(b) for b in self.boxes_conners]
        self.boxes_areas = [b.area for b in self.boxes_polygons]
        print("Done calculate boxes infor")

    def load_full_boxes_wh(self):
        boxes_wh = []
        for sample_id in self.sample_id_list:
            targets = self.load_targets(sample_id)
            for target in targets:
                cls, x, y, w, l, im, re = target
                if self.use_yaw_label:
                    yaw = np.arctan2(im, re)
                else:
                    yaw = 0
                boxes_wh.append([int(w * self.img_size), int(l * self.img_size), yaw])
        return np.array(boxes_wh)

    def cvt_box_2_polygon(self, box):
        return Polygon([(box[i, 0], box[i, 1]) for i in range(4)])

    def compute_iou(self, i):
        box_polygon, box_area = self.boxes_polygons[i], self.boxes_areas[i]
        intersections = [box_polygon.intersection(clus_polygon).area for clus_polygon in self.cluster_polygons]
        iou = [inter / (box_area + area2 - inter + 1e-12) for (area2, inter) in zip(self.cluster_areas, intersections)]
        # print('done compute_iou')
        return np.array(iou, dtype=np.float32)

    def avg_iou(self):
        return np.mean([np.max(self.compute_iou(i)) for i in range(self.num_boxes)])

    def kmeans(self, num_anchors):
        # The position of each point in each box
        distance = np.empty((self.num_boxes, num_anchors))

        # Last cluster position
        last_clu = np.zeros((self.num_boxes,))

        np.random.seed(0)

        # Randomly select k cluster centers
        self.cluster = self.boxes_wh[np.random.choice(self.num_boxes, num_anchors, replace=False)]
        self.cluster[:, 2] = 0  # Choose yaw = 0

        # cluster = random.sample(self.num_boxes, k)
        self.loop_cnt = 0
        while True:
            self.loop_cnt += 1

            print('\nThe new cluster of count {} is below:\n'.format(self.loop_cnt))
            for clus_ in self.cluster:
                w_, h_, yaw_ = clus_
                print('[{}, {}, {:.0f}],'.format(int(w_), int(h_), yaw_))

            self.cluster_conners = np.array(
                [kitti_bev_utils.get_corners(0, 0, clus[0], clus[1], clus[2]) for clus in self.cluster])
            self.cluster_polygons = [self.cvt_box_2_polygon(clus) for clus in self.cluster_conners]
            self.cluster_areas = [clus.area for clus in self.cluster_polygons]
            # Calculate the iou situation at five points from each line.
            for i in range(self.num_boxes):
                distance[i] = 1 - self.compute_iou(i)

            # Take out the smallest point
            near = np.argmin(distance, axis=1)

            if (last_clu == near).all():
                break

            # Find the median of each class
            for j in range(num_anchors):
                self.cluster[j] = np.median(self.boxes_wh[near == j], axis=0)
            self.cluster[:, 2] = 0  # Choose yaw = 0

            last_clu = near

    def load_targets(self, sample_id):
        """Load images and targets for the training and validation phase"""

        objects = self.get_label(sample_id)
        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)
        # on image space: targets are formatted as (class, x, y, w, l, sin(yaw), cos(yaw))
        targets = kitti_bev_utils.build_yolo_target(labels)

        return targets

    def remove_invalid_idx(self, image_idx_list):
        """Discard samples which don't have current training class objects, which will not be used for training."""

        sample_id_list = []
        for sample_id in image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)
            if not noObjectLabels:
                labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                                   calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_point_cloud_range(labels[i, 1:4]):
                        valid_list.append(labels[i, 0])

            if len(valid_list) > 0:
                sample_id_list.append(sample_id)

        return sample_id_list

    def check_point_cloud_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return kitti_data_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(label_file)
        return kitti_data_utils.read_label(label_file)


if __name__ == '__main__':
    dataset_dir = '../../dataset/kitti'
    num_anchors = 9
    img_size = 608
    use_yaw_label = True
    anchors_solver = Find_Anchors(dataset_dir, img_size, use_yaw_label=use_yaw_label)

    # Use k clustering algorithm
    anchors_solver.kmeans(num_anchors)
    areas = anchors_solver.cluster[:, 0] * anchors_solver.cluster[:, 1]
    anchors_solver.cluster = anchors_solver.cluster[np.argsort(areas)]
    print('Selected anchors_solver.cluster: ', end='')
    for clus in anchors_solver.cluster:
        w_, h_, yaw_ = clus
        print('{}, {}, {:.0f}'.format(int(w_), int(h_), yaw_), end=', ')
    print('avg_iou score: {:.2f}%'.format(anchors_solver.avg_iou() * 100))

    ############## *************************************************** #############
    #############                RESULTS                               #############
    ############## *************************************************** #############

    #######################################################################################
    ########### Selected anchors (use_yaw_label=True), fix yaw of anchor = 0 ##############
    #######################################################################################
    # 9 anchors: 11, 15, 0, 10, 24, 0, 11, 25, 0, 23, 49, 0, 23, 55, 0, 24, 53, 0, 24, 60, 0, 27, 63, 0, 29, 74, 0, avg_iou score: 46.01%
    # 6 anchors: 11, 15, 0, 11, 25, 0, 23, 49, 0, 23, 55, 0, 24, 53, 0, 25, 61, 0, avg_iou score: 45.82%

    #######################################################################################
    ######################## Finding by using (use_yaw_label=True) #######################
    #######################################################################################
    # anchors_solver.cluster = np.array([[10, 24, 1.55],
    #                                    [11, 14, 0.02],
    #                                    [11, 25, 0.48],
    #                                    [22, 46, 1.57],
    #                                    [23, 54, 1.57],
    #                                    [23, 50, 0.75],
    #                                    [24, 54, 0.02],
    #                                    [25, 60, 0.83],
    #                                    [28, 71, 1.56]])
    # if not use_yaw_label:
    #     anchors_solver.cluster[:, 2] = 0.
    # IoU score: 90.52% (use_yaw_label=False), 81.35% (use_yaw_label=True)

    #######################################################################################
    ######################## Finding by using (use_yaw_label=False) #######################
    #######################################################################################

    # anchors_solver.cluster = np.array(
    #     [[9, 24, 0],
    #      [11, 14, 0],
    #      [12, 25, 0],
    #      [22, 53, 0],
    #      [22, 44, 0],
    #      [23, 49, 0],
    #      [23, 56, 0],
    #      [24, 57, 0],
    #      [28, 72, 0]]
    # )
    # IoU score: 90.92% (use_yaw_label=False), and 47.07% (use_yaw_label=True)

    #######################################################################################
    ############################ Test Complex-YOLOv3 anchors ##############################
    #######################################################################################

    # anchors_solver.cluster = np.array([
    #     [11, 14, -np.pi],
    #     [11, 14, 0],
    #     [11, 14, np.pi],
    #     [11, 25, -np.pi],
    #     [11, 25, 0],
    #     [11, 25, np.pi],
    #     [23, 51, -np.pi],
    #     [23, 51, 0],
    #     [23, 51, np.pi]
    # ])
    # IoU Score: 84.24% (use_yaw_label=False), 44.94% (use_yaw_label=True)

    # Common below code should be uncomment if test with use_yaw_label option

    # anchors_solver.cluster_conners = np.array(
    #     [kitti_bev_utils.get_corners(0, 0, clus[0], clus[1], clus[2]) for clus in anchors_solver.cluster])
    # anchors_solver.cluster_polygons = [anchors_solver.cvt_box_2_polygon(clus) for clus in anchors_solver.cluster_conners]
    # anchors_solver.cluster_areas = [clus.area for clus in anchors_solver.cluster_polygons]

    # anchors_solver.cluster = anchors_solver.cluster[np.argsort(anchors_solver.cluster[:, 0])]
    # print('anchors_solver.cluster: {}'.format(anchors_solver.cluster))
    # print('acc:{:.2f}%'.format(anchors_solver.avg_iou() * 100))

    # anchors = 11, 14, 0.02,   10, 24, 1.55, 11, 25, 0.48,    22, 46, 1.57,   23, 50, 0.75,  23, 54, 1.57, 24, 54, 0.02,   25, 60, 0.83,   28, 71, 1.56
    # anchors = 11, 14, 0,   10, 24, 0,   11, 25, 0,    22, 46, 0,   23, 50, 0,  23, 54, 0,   24, 54, 0,   25, 60, 0,   28, 71, 0
